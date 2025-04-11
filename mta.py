import torch 
import torch.nn as nn 
import math
from typing import Tuple, Literal
import torch.nn.functional as F

class MultiTokenAttention(nn.Module): 

    def __init__(self, kernel_size:  Tuple[int,int, int], conv_option: Literal['pre','post','mix'],
                 embed_dim ,kv_dim, context_length, num_heads, droprate=0.0):  
        
        '''
        kernel_size: (c_q, c_k, c_h) kernel size for the query-key and head-mixing convolutions. 
        conv_options: The convolution option to mix. 
            - 'pre' = Both query-key convolution and head mixing convolution is done before applying softmax 
            - 'post' = Both convolutions are done after applying softmax. 
            - 'mix' = Query-key convolution is done pre-softmax and head-mixing convolution is done after applying softmax. 
        '''

        super(MultiTokenAttention, self).__init__() 
        
        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads!"

        self.W_q = nn.Linear(embed_dim, embed_dim) 
        self.W_k = nn.Linear(kv_dim, embed_dim) 
        self.W_v = nn.Linear(kv_dim, embed_dim)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1) 
        )

        self.context_length = context_length
        self.kv_dim = kv_dim
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.c_q = kernel_size[0] 
        self.c_k = kernel_size[1] 
        self.c_h = kernel_size[2] 
        self.conv_option = conv_option

        if conv_option in ['pre', 'post']: 
            self.conv = nn.Conv3d(
                in_channels=self.num_heads, 
                out_channels=self.num_heads, 
                kernel_size=(1, self.c_q, self.c_k), 
                stride=(1,1,1), # Non-overlapping convolution over head dimension 
                padding=0, # Asymmetric padding required, implemented in forward function
                groups = self.num_heads // self.c_h, 
                bias = False
            )
        
        elif conv_option == 'mix': 
            self.conv_qk = nn.Conv2d(
                in_channels=self.num_heads, 
                out_channels=self.num_heads, 
                kernel_size=(self.c_q, self.c_k), 
                groups=self.num_heads, # Depth-wise convolution; each head will have its own kernel. Input head convolutions should not mix to give output head channels.
                padding = 0, # Asymmetric padding implemented at runtime
                bias = False 
            )

            self.conv_heads = nn.Conv1d(
                in_channels=self.num_heads, 
                out_channels=self.num_heads, 
                kernel_size= 1, 
                groups= self.num_heads // self.c_h, 
                padding = 0, 
                bias = False
            )

        self.dropout = nn.Dropout(droprate) 
        self.proj_out = nn.Linear(embed_dim, embed_dim) 

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor: 
        
        assert len(query.shape) == 3, "Query should have shape (B,N,embed_dim)" 
        assert len(key.shape) == 3, "Key should have shape (B,N,kv_dim)" 
        assert len(value.shape) == 3, "Value should have shape (B,N,kv_dim)" 
        assert query.shape[-1] == self.embed_dim, f"Query should have embedding size {self.embed_dim}"
        assert key.shape[-1] == self.kv_dim, f"Key should have embedding size {self.kv_dim}"
        assert value.shape[-1] == self.kv_dim, f"Value should have embedding size {self.kv_dim}"
        
        B, N_q, _ = query.shape 
        _, N_kv, _ = key.shape 

        # Compute Q,K,V
        query = self.W_q(query)  # (B,N_q,embed_dim) 
        key = self.W_k(key)  # (B,N_kv, embed_dim) 
        value = self.W_v(value) 

        # Split dimensions into heads 
        query = query.view(B, N_q, self.num_heads, self.head_dim).transpose(1,2) # (B,H, N_q, d_h) 
        key = key.view(B, N_kv, self.num_heads, self.head_dim).transpose(1,2) # (B,H, N_kv, d_h) 
        value = value.view(B, N_kv, self.num_heads, self.head_dim).transpose(1,2) # (B,H, N_kv, d_h) 

        # Calculate attention scores 
        attention_scores = query @ key.transpose(2,3) # (B,H,N_q,N_kv) 

        # Scale scores 
        attention_scores = attention_scores / math.sqrt(self.head_dim) 

        # Pre-softmax convolution on both query-key and head mixing 
        if self.conv_option == 'pre':

            # Mask 1: Zeros above main diagonal 
            attention_scores.masked_fill_(self.mask.bool()[:N_q, :N_kv], 0) 

            # Add channel dimension for 3D conv 
            attention_scores = attention_scores.unsqueeze(2) # (B,H,1,N_q,N_kv) 

            # Asymmetric Padding 
            attention_scores = F.pad(attention_scores, (
                self.c_k//2, self.c_k - self.c_k//2 - 1, 
                self.c_q - 1, 0, 
                0,0
            ))

            # 3D Convolution on K-Q + head-mixing 
            out_conv = self.conv(attention_scores) # (B, H, 1, N_q, N_kv) 
            out_conv = out_conv.squeeze(2) # (B,H,N_q,N_kv) 

            # Mask 2: -Inf above main diagonal so that softmax ignores contribution 
            out_conv.masked_fill_(self.mask.bool()[:N_q, :N_kv], -torch.inf) 
            
            # Softmax 
            attention_weights = torch.softmax(out_conv, dim=-1) 

        # Post-softmax convolution on both query-key and head mixing
        elif self.conv_option == 'post': 
        
            # Mask 1: -Inf above main diagonal 
            attention_scores.masked_fill_(self.mask.bool()[:N_q, :N_kv], -torch.inf) 
            attention_weights = torch.softmax(attention_scores, dim=-1) 

            # Add channel dimension for 3D conv 
            attention_scores = attention_scores.unsqueeze(2) # (B,H,1,N_q,N_kv) 

            # Asymmetric Padding 
            attention_weights = F.pad(attention_weights, (
                self.c_k//2, self.c_k - self.c_k//2 - 1, 
                self.c_q - 1, 0, 
                0,0
            )) 

            # 3D Convolution on K-Q + head-mixing 
            out_conv = self.conv(attention_weights) # (B, H, 1, N_q, N_kv) 
            out_conv = out_conv.squeeze(2) # (B,H,N_q,N_kv) 

            # Mask 2: 0 above main diagonal so that future values are ignored 
            out_conv.masked_fill_(self.mask.bool()[:N_q, :N_kv], 0) 
            attention_weights = out_conv 
        
        # Pre-softmax convolution on query-key and post-softmax on head mixing 
        elif self.conv_option == 'mix': 
            
            # Mask 1: Zeros above main diagonal 
            attention_scores.masked_fill_(self.mask.bool()[:N_q, :N_kv], 0) 

            # Asymmetric Padding 
            attention_scores = F.pad(attention_scores, (
                self.c_k//2, self.c_k - self.c_k//2 - 1, 
                self.c_q - 1, 0, 
            ))

            # 2D Convolution on K-Q pre-softmax
            out_conv_qk = self.conv_qk(attention_scores) # (B, H, N_q, N_kv) 
            
            # Mask 2: -Inf above main diagnal before softmax 
            attention_weights = torch.softmax(out_conv_qk, dim=-1) 
            
            # Reshape for input to 1D conv 
            attention_weights = attention_weights.permute(0,2,3,1) # (B, N_q, N_kv, H)
            attention_weights = attention_weights.reshape(-1, self.num_heads, 1) # (B*N_q*N_kv, H, 1) 
            
            # 1D conv for head mixing 
            attention_weights = self.conv_heads(attention_weights) # (B*N_q*N_kv,H,1) 

            # Reshape back 
            attention_weights = attention_weights.reshape(B,N_q, N_kv, self.num_heads, 1) # (B,N_q,N_kv,H) 
            attention_weights = attention_weights.permute(0,3,1,2) # (B,H,N_q,N_kv) 
            

        # Calculate context vector 
        context_vector = attention_weights @ value # (B,H, N_q, d_h) 
        context_vector = context_vector.transpose(1,2).view(B,N_q, self.embed_dim) 

        # Feed Forward 
        context_vector = self.proj_out(context_vector) 

        return context_vector