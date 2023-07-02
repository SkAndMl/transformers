from torch import nn
import torch
import math

from typing import Tuple

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, 
                 d_model: int=512,
                 num_heads: int=8):
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model%num_heads == 0, f"Dimensionality of the model {d_model} should be divisible by the number of heads {num_heads}"

        self.W_v = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def scaled_dot_product_attention(query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # query = key = value -> [batch, query_len/key_len/value_len, num_heads, d_k]
        d_k = query.size(-1)
        compatability = torch.einsum("bqhd,bkhd->bhqk", [query, key]) # compatability -> [batch, num_heads, query_len, key_len]
        if mask is not None:
            compatability = compatability.masked_fill(mask==0, -float("inf"))
        compatability = torch.softmax(compatability, dim=-1)/math.sqrt(d_k)
        x = torch.einsum("bhqk,bkhd->bqhd", [compatability, value]) # x -> [batch, query_len, num_heads, d_k]
        return x, compatability
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None):
        
        b, query_len, key_len, d_k = query.shape[0], query.shape[1], key.shape[1], self.d_model//self.num_heads 
        query, key, value = self.W_q(query), self.W_k(key), self.W_v(value)
        query = query.view(b, query_len, self.num_heads, d_k)
        key = key.view(b, key_len, self.num_heads, d_k)
        value = value.view(b, key_len, self.num_heads, d_k)
        x, _ = MultiHeadAttentionLayer.scaled_dot_product_attention(query, key, value, mask)
        return self.W_o(x.reshape(b, query_len, self.d_model))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8,
                 dropout: float=0.1) -> None:
        super().__init__()
        self.multihead_attention = MultiHeadAttentionLayer(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        return self.layer_norm(query + self.dropout(self.multihead_attention(query, key, value, mask)))
    

class FeedForwardBlock(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 d_ff: int=2048,
                 dropout: float=0.1) -> None:
        
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(self.fc(x)))