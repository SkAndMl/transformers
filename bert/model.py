import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import sys
sys.path.insert(0, '..')
from scripts.scripts import Embedding, Encoder, create_padding_mask

class BERTMaskedLM(nn.Module):

    PAD_TOKEN = 0
    CLS_TOKEN = 1
    SEP_TOKEN = 2
    MASK_TOKEN = 3
    UNK_TOKEN = 4

    def __init__(self,
                 config,
                 vocab_size: int=1000) -> None:
        super().__init__()
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_encoders"]
        self.embedding = Embedding(config=config,
                                   vocab_size=vocab_size)
        self.bert = Encoder(config=config)
        self.masked_lm = nn.Linear(in_features=self.d_model,
                                      out_features=vocab_size)
        
    

    def __repr__(self) -> str:
        return f"BERT(num_layers={self.n_layers}, d_model={self.d_model}, num_heads={self.n_heads})"
    
    def __str__(self) -> str:
        return f"BERT(num_layers={self.n_layers}, d_model={self.d_model}, num_heads={self.n_heads})"

    def forward(self,
                x: torch.Tensor,
                masked_tokens: torch.Tensor,
                masked_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        mask = create_padding_mask(batch=x, padding_idx=self.PAD_TOKEN)

        # x -> [B, S]
        x = self.embedding(x) # [B, S, D_MODEL]
        x = self.bert(x, mask) # [B, S, D_MODEL]
        x = x[range(len(masked_idx)), masked_idx, :].squeeze() # B, D_MODEL
        logits = self.masked_lm(x) # B, VOCAB_SIZE

        loss = F.cross_entropy(logits, masked_tokens.squeeze())

        return logits, loss
        
