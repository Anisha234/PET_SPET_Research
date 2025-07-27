import torch
import torch.nn as nn

class CTPAModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers,drop_prob = 0.0, max_len=1024):
        super(CTPAModel, self).__init__()
        
        self.cls_emb  = torch.nn.Parameter(torch.randn(1, 1,embed_dim))
        #self.cls_emb.requires_grad_(False)  
        self.embed_dim = embed_dim
          # Positional embedding: [max_len, embed_dim]
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        
        self.output_linear = nn.Linear(self.embed_dim, 2)
        self.dropout = nn.Dropout(0.1)
        self.drop_prob = drop_prob
        
    
    def forward(self, x):
        # print(len(x_list), x_list[0].size()) #2 x 3 x 512 x 512
        
         
        if self.training:
            b,s,d = x.size()
            keep_mask = torch.rand(s) > self.drop_prob  # shape: (s,)
            
        #    print(x.size())
            # Apply mask across sequence dimension
            x = x[:, keep_mask, :]  # shape: (b, s', d) where s' <= s
        #    print("Mask",keep_mask)
        #    print(x.size())
        
        cls_tokens = self.cls_emb.expand(x.size()[0], -1, -1)  # (batch_size, 1, embed_dim)
        #x = self.dropout(x)
        x = torch.cat((cls_tokens, x), dim=1)
        B,L,D = x.size()
        # Add positional embedding
        #
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # → (1, L)
        pos_emb = self.pos_embedding(positions)      # → (1, L, embed_dim)

        #x = x + pos_emb                              # broadcasting over batch 
            
        #print(x.size())
       #x = torch.cat(features, dim=1)
        x = self.encoder(x) #Transformer Encoder b x 78 x 512
        
        #x = x.mean(dim=(1), keepdim=False)
        x = x[:,0,:]
        return self.output_linear(x), x    