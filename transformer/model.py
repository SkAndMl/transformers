import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
import sys
sys.path.insert(0, "..")
from scripts.scripts import Embedding, Encoder, Decoder, create_causal_mask, create_padding_mask

device = "mps" if torch.backends.mps.is_available() else "cpu"

class TransformerForSeqToSeq(nn.Module):

    SOS_TOKEN = 1
    EOS_TOKEN = 2

    def __init__(self, config, 
                 src_vocab_size: int=1000, tgt_vocab_size: int=1000, padding_idx: int=0) -> None:
        
        super().__init__()

        self.padding_idx = padding_idx
        self.config = config

        self.src_embedding = Embedding(config=config, vocab_size=src_vocab_size)
        self.tgt_embedding = Embedding(config=config, vocab_size=tgt_vocab_size)    
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)
        self.cls_head = nn.Sequential(
            nn.Dropout(p=config["dropout"]),
            nn.Linear(in_features=config["d_model"],
                      out_features=tgt_vocab_size)
        )
    
    def forward(self, 
                src: torch.Tensor,
                tgt: torch.Tensor=None) -> torch.Tensor:

        B, S = src.shape
 
        padding_mask = create_padding_mask(src, padding_idx=self.padding_idx).to(device=device) # B, 1, S

        src_input = self.src_embedding(src) # B, S, D_MODEL
        encoder_output = self.encoder(src_input) # B, S, D_MODEL

        if tgt is not None:
            
            _, T = tgt.shape
            causal_mask = create_causal_mask(T).to(device=device)

            tgt_input = self.tgt_embedding(tgt) # B, T, D_MODEL
            decoder_output = self.decoder(tgt_input, encoder_output, causal_mask, padding_mask) # B, T, D_MODEL
            logits = self.cls_head(decoder_output) # B, T, VOCAB_SIZE
            loss = F.cross_entropy(logits.reshape(B*T, -1), tgt.reshape(-1))
            return logits, loss
        else:
            translation = torch.tensor([[self.SOS_TOKEN]], device=device) # 1, 1
            while translation.shape[-1] < self.config["context_length"] or translation[-1, -1].item()!=self.EOS_TOKEN:
                decoder_output = self.decoder(self.tgt_embedding(translation), encoder_output, None, padding_mask) # 1, T+1, D_MODEL
                logits = self.cls_head(decoder_output).squeeze() # T+1, VOCAB_SIZE
                next_word = torch.argmax(logits, dim=-1)[-1:, :] # 1, 1
                translation = torch.cat((translation, next_word), dim=-1)
            return translation
    
    def translate(self,
                  src: torch.Tensor) -> torch.Tensor:
        
        # src -> B, S
        src_input = self.src_embedding(src)
        return self(src_input, None)






if __name__ == '__main__':
    import json
    with open("../config/base_config.json") as f:
        config = json.load(f)
    t = TransformerForSeqToSeq(config=config)
    src = torch.arange(start=0, end=6, dtype=torch.long).reshape(shape=(3, 2)).to(device=device)
    tgt = torch.arange(start=10, end=16, dtype=torch.long).reshape(shape=(3, 2)).to(device=device)
    summary(model=t, input_data=[src, tgt])