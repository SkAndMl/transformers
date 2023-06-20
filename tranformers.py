import torch
from torch import nn
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt


class Embedding(nn.Module):
    
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int=512):
        
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                           embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        emb = self.embedding(x)
        pos_enc = torch.zeros(size=(x.size()[0], self.embedding_dim),
                             dtype=torch.float)
        
        for pos in range(0, x.size()[0]):
            for i in range(self.embedding_dim):
                if i%2==0:
                    pos_enc[pos, i] = torch.sin(torch.tensor(pos/10000**(2*i/self.embedding_dim)))
                else:
                    pos_enc[pos, i] = torch.cos(torch.tensor(pos/10000**(2*i/self.embedding_dim)))
        
        return emb + pos_enc
    
class AttentionLayer(nn.Module):
    
    def __init__(self,
                 d_model: int=512):
        super().__init__()
        self.d_model = d_model
    
    def forward(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               mask: bool=False) -> torch.Tensor: 
        return self._scaled_dot_product_attention(query, key, value, mask)
    
    def _scaled_dot_product_attention(self,
                                     query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     mask: bool=False) -> torch.Tensor:
        
        dot_product = torch.matmul(query, key.transpose(0, 1))
        if mask:
            dot_product = dot_product.tril()
        scaled_dot_product = dot_product/torch.sqrt(torch.tensor(self.d_model))
        softmax_scaled = torch.softmax(scaled_dot_product, dim=-1)
        return torch.matmul(softmax_scaled, value)
    

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self,
                d_model: int=512,
                num_heads: int=8) -> None:
        
        super().__init__()
        
        assert d_model%num_heads==0, f"d_model {d_model} should be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model//num_heads
        self.W_out = nn.Linear(in_features=d_model, 
                           out_features=d_model)
        
        self.attention_heads = nn.ModuleList([AttentionLayer(d_model=d_model) for _ in range(num_heads)])
    
    def forward(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               mask: bool=False) -> torch.Tensor:
        
        attn_outputs = [self.attention_heads[i](query, key, value, mask) for i in range(self.num_heads)]
        attn_output = torch.cat(attn_outputs, dim=1)
        return self.W_out(attn_output)
    

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self,
                d_model: int=512,
                num_heads: int=8) -> None:
        
        super().__init__()
        self.mha = MultiHeadAttentionLayer(d_model=d_model,
                                          num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.key_dim = d_model//num_heads
        self.Wq = nn.Linear(in_features=d_model, out_features=self.key_dim)
        self.Wk = nn.Linear(in_features=d_model, out_features=self.key_dim)
        self.Wv = nn.Linear(in_features=d_model, out_features=self.key_dim)
    
    def forward(self, 
                x:torch.Tensor,
                mask: bool=False) -> torch.Tensor:
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        return self.layer_norm(x + self.mha(query, key, value, mask))
    

class FeedForwardBlock(nn.Module):
    
    def __init__(self,
                d_model: int=512,
                inner_dim: int=2048) -> None:
        
        super().__init__()
        self.ff_layer = nn.Sequential(
            nn.Linear(in_features=d_model,
                     out_features=inner_dim),
            nn.ReLU(),
            nn.Linear(in_features=inner_dim,
                     out_features=d_model)
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.ff_layer(x))

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8,
                 inner_dim: int=2048):
        
        super().__init__()
        self.mha_block = MultiHeadAttentionBlock(d_model=d_model,
                                                 num_heads=num_heads)
        self.fc_block = FeedForwardBlock(d_model=d_model,
                                         inner_dim=inner_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_block(self.mha_block(x))

class TransformerEncoder(nn.Module):

    def __init__(self,
                 num_encoders: int=6,
                 d_model: int=512,
                 num_heads: int=8,
                 inner_dim: int=2048) -> None:
        
        super().__init__()
        self.num_encoders = num_encoders
        self.encoder_blocks = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, inner_dim) for _ in range(num_encoders)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = x
        for i in range(self.num_encoders):
            op = self.encoder_blocks[i](op)
        return op
    
class EncoderDecoderMultiHeadAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8):
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model//num_heads

        self.Wq = nn.Linear(in_features=d_model, out_features=self.key_dim)
        self.Wk = nn.Linear(in_features=d_model, out_features=self.key_dim)
        self.Wv = nn.Linear(in_features=d_model, out_features=self.key_dim)

        self.mha = MultiHeadAttentionLayer(d_model=d_model,
                                           num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self,
                encoder_outputs: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        query = self.Wq(x)
        key = self.Wk(encoder_outputs)
        value = self.Wv(encoder_outputs)
        return self.layer_norm(x + self.mha(query, key, value))
    
class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8,
                 inner_dim: int=2048) -> None:
        super().__init__()
        self.edmha = EncoderDecoderMultiHeadAttentionBlock(d_model, num_heads)
        self.mha = MultiHeadAttentionBlock(d_model, num_heads)
        self.fc_block = FeedForwardBlock(d_model, inner_dim)
    
    def forward(self, 
                encoder_outputs: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        
        op = self.mha(x, True)
        op = self.edmha(encoder_outputs, op)
        op = self.fc_block(op)
        return op

class TransformerDecoder(nn.Module):

    def __init__(self,
                 num_decoders: int=8,
                 d_model: int=512,
                 num_heads: int=8,
                 inner_dim: int=2048) -> None:
        
        super().__init__()
        self.num_decoders = num_decoders
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, inner_dim) for _ in range(num_decoders)])
    
    def forward(self,
                encoder_outputs: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        
        op = x
        for i in range(self.num_decoders):
            op = self.decoder_layers[i](encoder_outputs, op)
        return op

class Transformer(nn.Module):

    def __init__(self,
                 input_embedding_dim: int=1000,
                 output_embedding_dim: int=1000,
                 num_encoders: int=6,
                 num_decoders: int=6,
                 d_model: int=512,
                 num_heads: int=8,
                 inner_dim: int=2048) -> None:
        
        super().__init__()
        self.input_embedding_layer = Embedding(num_embeddings=input_embedding_dim,
                                         embedding_dim=d_model)
        self.output_embedding_layer = Embedding(num_embeddings=output_embedding_dim,
                                                embedding_dim=d_model)
        self.encoder_layer = TransformerEncoder(num_encoders, d_model, num_heads, inner_dim)
        self.decoder_layer = TransformerDecoder(num_decoders, d_model, num_heads, inner_dim)
        self.fc_out = nn.Linear(in_features=d_model, out_features=output_embedding_dim)
    
    def forward(self,
                input_sentence: torch.Tensor,
                output_sentence: torch.Tensor) -> torch.Tensor:
        
        input_embedding = self.input_embedding_layer(input_sentence)
        output_embedding = self.output_embedding_layer(output_sentence)
        encoder_outputs = self.encoder_layer(input_embedding)
        decoder_outputs = self.decoder_layer(encoder_outputs, output_embedding)
        outputs = torch.softmax(self.fc_out(decoder_outputs), dim=1)
        return outputs