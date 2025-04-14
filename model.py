import torch 
import torch.nn as nn 
from baseline_attentions import MultiHeadAttention
from mta import MultiTokenAttention
from utils import GELU

class GPTModel(nn.Module): 
    def __init__(self, cfg: dict, att_mechanism: str): 
        super(GPTModel, self).__init__() 
        self.use_kv_cache = False 
        self.context_length = cfg["context_length"]

        # Token and position embedding layers 
        self.token_emb_layer = nn.Embedding(cfg["vocab_size"], cfg["token_emb_dim"]) 
        self.pos_emb_layer = nn.Embedding(cfg["context_length"], cfg["token_emb_dim"])  

        # Dropout layer for generated input embeddings 
        self.drop_inp_emb = nn.Dropout(cfg["droprate"])

        # Transformer blocks 
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, att_mechanism) for _ in range(cfg["num_layers"])]
        )

        # Layer norm 
        self.final_norm = LayerNorm(cfg["token_emb_dim"])

        # Output prediction head 
        self.out_head = nn.Linear(cfg["token_emb_dim"], cfg["vocab_size"]) 
    
    def forward(self, x, start_pos: int = None):
        """
        x: Tokenized text. Will have shape (B, num_tokens) 
        output: (B, num_tokens, vocab_size) 
        """   
        assert len(x.shape) == 2, "Input must be of shape (B, num_tokens)"
        B, num_tokens = x.shape 

        token_embeddings = self.token_emb_layer(x) # (B, num_tokens, token_emb_dim) 
        
        if self.use_kv_cache and num_tokens == 1:  # KV cache must add the position embedding according to current position in generated sequence. Otherwise, we always add PE of 0th position. 
            pos_embeddings = self.pos_emb_layer(torch.tensor([start_pos], device=x.device)) 
        else: 
            pos_embeddings = self.pos_emb_layer(torch.arange(num_tokens, device=x.device)) 

        input_embeddings = token_embeddings + pos_embeddings 

        # Dropout on input embeddings 
        input_embeddings = self.drop_inp_emb(input_embeddings)  

        # Pass through transformer blocks 
        out = input_embeddings
        for i,trf_block in enumerate(self.trf_blocks): 
            out = trf_block(out, start_pos)  

        # Pass through layer norm 
        out = self.final_norm(out) 

        # Pass through prediction head 
        out = self.out_head(out) 

        return out 
    
    def toggle_kv_cache(self, use_kv_cache: bool):
        """Dynamically enable/disable KV cache in all transformer blocks"""
        self.use_kv_cache = use_kv_cache
        for block in self.trf_blocks:
            block.use_kv_cache = use_kv_cache

class TransformerBlock(nn.Module): 
    def __init__(self, cfg: dict, att_mechanism: str): 
        super(TransformerBlock, self).__init__()  
        self.layer_norm1 = LayerNorm(cfg["token_emb_dim"]) 
        self.use_kv_cache = False
        
        # Multi Head Attention
        if att_mechanism == 'mha': 
            self.att = MultiHeadAttention(
                cfg["token_emb_dim"], 
                cfg["token_emb_dim"], 
                cfg["droprate"], 
                cfg['max_batch_size'], 
                cfg['max_seq_len'],
                cfg["num_heads"], 
                cfg["qkv_bias"] 
                ) 
        
        # Multi Token Attention
        elif att_mechanism == 'mta': 
            self.att = MultiTokenAttention(
                cfg['kernel_size'], 
                cfg['conv_option'], 
                cfg['token_emb_dim'], 
                cfg['kv_dim'], 
                cfg['num_heads'], 
                cfg['max_batch_size'], 
                cfg['max_seq_len'],
                cfg["droprate"]
            )
        
        self.dropout = nn.Dropout(cfg["droprate"]) 
        self.layer_norm2 = LayerNorm(cfg["token_emb_dim"]) 
        self.ff = FeedForward(cfg["token_emb_dim"]) 

    def forward(self, x, start_pos: int = None, is_causal=True): 

        self.att.kv_cache_enabled = self.use_kv_cache and not self.training  
        if self.att.kv_cache_enabled: 
            assert start_pos is not None, "Must provide start_pos during inference for using KV Cache!" 
            
        res = x # First res connection
        out = self.layer_norm1(x) # (B,N,token_emb) 
        out = self.att(out, start_pos, is_causal) # (B,N, token_emb) 
        out = self.dropout(out) # (B, N, token_emb) 
        out = out + res # Res connection # (B,N, token_emb) 

        res = out # Second res connection
        out = self.layer_norm2(out) # (B, N, token_emb) 
        out = self.ff(out) # (B, N, token_emb) 
        out = self.dropout(out) # (B,N,token_emb) 
        out = out + res # Res connection # (B,N,token_emb) 

        return out 
    
class LayerNorm(nn.Module): 
    def __init__(self, emb_dim: int, eps=1e-5): 
        super(LayerNorm, self).__init__() 
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 
        self.eps = eps 

    def forward(self, x: torch.Tensor): 
        # x has shape (B, N, emb_dim) 

        mean =  x.mean(-1, keepdim=True) # (B,N,1) 
        var = x.var(-1, keepdim=True,unbiased=False) # (B,N,1) Unbiased = False divides by constant "n" in variance formula instead of "n-1" (Bessel's Correction) 
        normalized_x = (x-mean)/torch.sqrt(var + self.eps) # (B,N,emb_dim) 

        return self.scale * normalized_x + self.shift  

class FeedForward(nn.Module):  # Feed Forward modules help a lot in model understanding and genearlization because it allows exploration of a richer representation since we are expanding dimensions
    def __init__(self, emb_dim): 
        super(FeedForward, self).__init__()
        self.ff1 = nn.Linear(emb_dim, 4*emb_dim) 
        self.act = GELU() 
        self.ff2 = nn.Linear(4*emb_dim, emb_dim)  

        self.layers = nn.Sequential(self.ff1, self.act, self.ff2)
     
    def forward(self, x): 
        return self.layers(x) 