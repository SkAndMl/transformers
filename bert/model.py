import torch
from torch import nn
from typing import Tuple
import sys
sys.path.insert(0, '..')
from scripts.scripts import TransformerEncoder

class BERT(nn.Module):
    def __init__(self,
                 num_layers: int=12,
                 d_model: int=768,
                 num_heads: int=12,
                 vocab_size: int=1000,
                 d_ff: int=2048,
                 attn_dropout: float=0.1,
                 ff_dropout: float=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bert_encoder = TransformerEncoder(num_encoders=num_layers,
                                               d_model=d_model,
                                               num_heads=num_heads,
                                               d_ff=d_ff,
                                               attn_dropout=attn_dropout,
                                               ff_dropout=ff_dropout)
        self.masked_block = nn.Linear(in_features=d_model,
                                      out_features=vocab_size)
        self.nsp_block = nn.Linear(in_features=d_model,
                                        out_features=2)
    

    def __repr__(self) -> str:
        return f"BERT(num_layers={self.num_layers}, d_model={self.d_model}, num_heads={self.num_heads})"
    
    def __str__(self) -> str:
        return f"BERT(num_layers={self.num_layers}, d_model={self.d_model}, num_heads={self.num_heads})"

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                masked_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
            x -> Input embedding, shape: [batch_size, max_seq_len, d_model]
            mask -> Mask for padding, shape: [batch_size, 1, 1, max_seq_len]
            masked_idx -> For each sequence a different index position has been masked
            and the encoder's contextual representation of the masked token will be
            used to predict the true token. To extract the representation for each 
            sequence the different index positions are passed in masked_idx which will 
            be used for indexing the representation. Shape: [batch_size]
        """
        x = self.bert_encoder(x, mask) # [batch_size, max_seq_len, d_model]
        masked_tokens = x[range(len(masked_idx)), masked_idx]

        # For NSP prediction the BERT paper uses the '[CLS]' token which is the 
        # 0th index in each sequence and it is accessed by indexing '0' along the
        # first dimension
        nsp_logits = self.nsp_block(x[:, 0, :]) # x[:, 0, :] -> [batch_size, d_model]
        masked_tokens_logits = self.masked_block(masked_tokens)
        return masked_tokens_logits, nsp_logits
        
