
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm


class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                embed_dim= 512,
                context_length= 77,
                vocab_size= 50257,
                transformer_width= 512,
                transformer_heads= 8,
                transformer_layers= 16,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        print("learning_rate",learning_rate)
        self.context_length = context_length
        self.encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
            )
        self.loss=torch.nn.CrossEntropyLoss(reduction='sum')

        self.vocab_size = vocab_size
        print("vocab_size",vocab_size)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)#
        self.geoCode_embedding = nn.Embedding(vocab_size, transformer_width)
        self.Location_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
       
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
  
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
 
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5
       
        for _,layer in self.encoder.named_modules():
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    nn.init.normal_(block.weight, std=1)
                    nn.init.zeros_(block.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=fc_std)
                nn.init.zeros_(layer.bias)
        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)

    def encode_text(self, text, geoCode, Location):

        # print("Location(x)",x.shape)
        x = self.token_embedding(text).type(self.dtype) +  self.geoCode_embedding(geoCode).type(self.dtype) + self.Location_embedding(Location).type(self.dtype) 
        x=x+ self.positional_embedding.type(self.dtype)
        x=x/4
        x = x.permute(1, 0, 2)  # NLD -> LND
        print("x",x.shape)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        return x


    # @torch.jit.script
    def forward(self, text, geoCode, Location):
        return self.encode_text(text, geoCode, Location)
        
        
        
        
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        text= batch["text"].squeeze(1)
        geoCode= batch["geonouns"].squeeze(1)
        Location= batch["plnames"].squeeze(1)
        #create mask for noise
        mask = torch.bernoulli(torch.full(text.shape, 0.15,device=self.device)).type(self.dtype)
        #randomly add noise Electra style...
        # print("mask",mask.shape)
        # print("maskmade",torch.randint_like(text,0,self.vocab_size,device=self.device)*mask)
        # print("text",text.shape)
        # print("input text",(text %self.vocab_size).shape)

        #+ (torch.randint_like(text,0,self.vocab_size,device=self.device)*mask),0,self.vocab_size)
        x1 = self(text, geoCode, Location)
        x2 = self(torch.clamp(text+(torch.randint_like(text,0,self.vocab_size,device=self.device)*mask).type(self.dtype),0, self.vocab_size), geoCode, Location)
        print("x1",x1.shape)
        print("x2",x2.shape)

        
        #add noise to x1 and x2
        x1 = x1 + (torch.randn_like(x1) * 0.05)
        x2 = x2 + (torch.randn_like(x2) * 0.05)
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)

        Lossx1=self.loss(x1 @ x2.T, torch.arange(x1.shape[0]))
        Lossx2=self.loss(x2 @ x1.T, torch.arange(x2.shape[0]))
        loss=Lossx1+Lossx2
        loss=loss/2
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    def on_train_epoch_end(self) -> None:
        #log embeddings for geonouns and plnames
        self.logger.experiment.add_embedding(self.Location_embedding.weight,tag="plnames")
        self.logger.experiment.add_embedding(self.geoCode_embedding.weight,tag="geonouns")
        self.logger.experiment.add_embedding(self.token_embedding.weight,tag="text")
        return super().on_train_epoch_end()
            
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=10e-8,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            )
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]
