
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm

import wandb
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
        self.loss=torch.nn.CrossEntropyLoss(reduction='mean')

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
        for _,layer in self.encode_image.named_modules():
            if isinstance(layer, nn.ModuleList):
                for block in layer:

                    nn.init.normal_(block.weight, std=1)
                    nn.init.zeros_(block.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1)
                nn.init.zeros_(layer.bias)
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

    def encode_text(self, text, geoCode, Location,eot):

        # print("Location(x)",x.shape)
        #make one hot of geoCode and Location by finding zeros and non zeros
        one_hot_geoCode = geoCode.bool().type(self.dtype).unsqueeze(-1)
        one_hot_Location = Location.bool().type(self.dtype).unsqueeze(-1)
        x = self.token_embedding(text).type(self.dtype) 
        x=x + torch.mul(one_hot_geoCode,self.geoCode_embedding(geoCode).type(self.dtype)) 
        x=x+ torch.mul(one_hot_Location,self.Location_embedding(Location).type(self.dtype))
        x=x+ self.positional_embedding.type(self.dtype)
        #print("x",x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(eot.shape[0]), eot] 
        return x


    # @torch.jit.script
    def forward(self, text, geoCode, Location,eot):
        return self.encode_text(text, geoCode, Location,eot)
        
        
        
        
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        text= batch["text"].squeeze(1)
        geoCode= batch["geonouns"].squeeze(1)
        Location= batch["plnames"].squeeze(1)
        labels=torch.arange(text.shape[0],device=self.device)
        eot=text.argmax(dim=-1)
        mask = torch.bernoulli(torch.full(text.shape, 0.05,device=self.device)).long()
        mask2= torch.bernoulli(torch.full(text.shape, 0.05,device=self.device)).long()
        x1 = self((text+ (torch.randint_like(text,0,self.vocab_size,device=self.device)*mask)) % self.vocab_size, geoCode, Location,eot)
        x2 = self((text+ (torch.randint_like(text,0,self.vocab_size,device=self.device)*mask2)) % self.vocab_size, geoCode, Location,eot)
              
        #add noise to x1 and x2
    
    
        # print("x1a",x1)
        
        # x1 = x1 + (torch.randn_like(x1) * 0.05)
        # x2 = x2 + (torch.randn_like(x2) * 0.05)
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)
        # print("x1b",x1)
        # print("x2",x2)
        l1= x1 @ x2.T
        l2= x2 @ x1.T
        # print("l1",l1)
        # print("l2",l2)
        Lossx1=self.loss( l1 *self.logit_scale.exp(), labels) # *self.logit_scale.exp()
        Lossx2=self.loss( l2 *self.logit_scale.exp(), labels)

        loss=Lossx1+Lossx2
        loss=loss/2
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def on_train_epoch_end(self) -> None:
        #log embeddings for geonouns and plnames as table

        self.logger.experiment.log(
            {"LocationEmbedding": wandb.Table(
                columns = ["Dimension {}".format(i) for i in range(self.Location_embedding.weight.shape[1])], 
                data    = self.Location_embedding.weight.detach().cpu().numpy()
                )})
        self.logger.experiment.log(
            {"geoCodeEmbedding": wandb.Table(
                columns = ["Dimension {}".format(i) for i in range(self.geoCode_embedding.weight.shape[1])], 
                data    = self.geoCode_embedding.weight.detach().cpu().numpy()
                )})
        self.logger.experiment.log({
            "textEmbedding": wandb.Table(
                columns = ["Dimension {}".format(i) for i in range(self.token_embedding.weight.shape[1])],
                data    = self.token_embedding.weight.detach().cpu().numpy()
                )
            })

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8,
            )
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]
