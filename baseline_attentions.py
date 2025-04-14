import torch 
import torch.nn as nn 
import math 

class MultiHeadAttention(nn.Module): 

    def __init__(self, inp_emb_dim, context_dim, dropout, max_batch_size, max_seq_len,  num_heads=4,qkv_bias=False):  
        super(MultiHeadAttention, self).__init__() 
        
        assert context_dim % num_heads == 0, "Required context_dim should be divisible by num_heads"
        
        self.inp_emb_dim = inp_emb_dim 
        self.context_dim = context_dim
        self.num_heads = num_heads  
        self.head_dim = context_dim // num_heads 
        self.W_query = nn.Linear(inp_emb_dim, context_dim, bias=qkv_bias) 
        self.W_key = nn.Linear(inp_emb_dim, context_dim, bias=qkv_bias) 
        self.W_value = nn.Linear(inp_emb_dim, context_dim, bias=qkv_bias) 
        self.kv_cache_enabled = False  
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.dropout = nn.Dropout(dropout) 
        self.proj_out = nn.Linear(context_dim, context_dim) # Used to merge the different heads. Optional but used in LLM architectures heavily. 

        self.register_buffer("cache_k", None) # Buffers for caches 
        self.register_buffer("cache_v", None)  

    def forward(self, inputs: torch.Tensor, start_pos: int = None, is_causal=True): 
        assert len(inputs.shape) == 3, "Input must be of shape (num_batches, num_tokens, token_dimensions)"  
        assert inputs.shape[-1] == self.inp_emb_dim, "Input hidden dimension must be equal to inp_emb_dim passed into MHA!"
        if self.kv_cache_enabled: 
            assert start_pos is not None, "Must provide start_pos argument if using kv-cache"

        B, num_tokens, _ = inputs.shape 
        
        # Create merged K,Q,V 
        K = self.W_key(inputs) # Shape = (B,N,context_dim) 
        Q = self.W_query(inputs) 
        V = self.W_value(inputs) 

        # Split K,Q,V into multiple heads 

        # Split context_dim into different heads 
        K = K.view(B, num_tokens, self.num_heads, self.head_dim) # Splits the last context_dim across heads. 
        Q = Q.view(B, num_tokens, self.num_heads, self.head_dim) 
        V = V.view(B, num_tokens, self.num_heads, self.head_dim) 

        if self.kv_cache_enabled: 
            if self.cache_k is None: 
                # Initialize cache for keys and values 
                self.cache_k = torch.empty(self.max_batch_size, self.max_seq_len, self.num_heads, self.head_dim, device=inputs.device)
                self.cache_v = torch.empty_like(self.cache_k, device=inputs.device) 
            
            # Cache keys and values 
            self.cache_k[:B, start_pos: start_pos + num_tokens] = K 
            self.cache_v[:B, start_pos: start_pos + num_tokens] = V 

            # Extract cached keys and values for computations 
            K = self.cache_k[:B, :start_pos + num_tokens] # (B, N', num_heads, head_dim) 
            V = self.cache_v[:B, :start_pos + num_tokens] 


        # Switch positions of num_heads so that batch matrix multiplication can be done across several heads in parallel 
        K = K.transpose(1,2) # (B,num_heads, N, head_dim) OR  (B, num_heads, N' , head_dim) if kv_Cache
        Q = Q.transpose(1,2) # (B,num_heads, N, head_dim) OR (B, num_heads, 1, head_dim) if kv_cache
        V = V.transpose(1,2)

        # Calculate attention weights 
        attention_scores = Q @ K.transpose(2,3) # (B, num_heads , N, N) or (B, num_heads, 1, N') if kv_cache 

        # Causal self attention only if KV cache is not being used or for initial input prompt using KV cache
        if (not self.kv_cache_enabled or attention_scores.shape[-2] != 1) and is_causal: 
            mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1) 
            attention_scores.masked_fill_(mask.bool(), -torch.inf) 

        attention_weights = torch.softmax(attention_scores/ math.sqrt(K.shape[-1]), dim=-1) # d_k is the dimension per head in this case

        # Dropout on attention weights 
        attention_weights = self.dropout(attention_weights)

        # Calculate context vectors 
        Z = attention_weights @ V # (B, num_heads, N, head_dim) OR (B, num_heads, 1 , head_dim) 
        Z = Z.transpose(1,2).contiguous() # (B, N, num_heads, head_dim) OR (B, 1, num_heads , head_dim)
        Z = Z.view(B, num_tokens, self.context_dim)  # (B, N, context_dim) OR (B,1, context_dim) 

        # Linearly project merged head information to get final context vector 
        context_vec = self.proj_out(Z) # (B, N, context_dim) OR (B,1, context_dim)

        return context_vec
    
