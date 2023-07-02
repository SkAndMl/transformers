from torch import nn
import torch
import math

class TransformerEmbedding(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 vocab_size: int=1000,
                 max_seq_len: int=10,
                 dropout: float=0.1):
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)
        self.pe = torch.zeros(max_seq_len, d_model, requires_grad=False)
        
        for i in range(max_seq_len):
            for j in range(d_model):
                if j%2==0:
                    self.pe[i, j] = math.sin(i/(10000**(2*j/d_model)))
                else:
                    self.pe[i, j] = math.cos(i/(10000**(2*j/d_model)))
        
        self.pe = self.pe.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [batch_size, seq_len]
        embed_out = self.embedding(x) # embed_out -> [batch_size, seq_len, ]
        return self.dropout(self.pe[:, embed_out.size(1), :] + embed_out)