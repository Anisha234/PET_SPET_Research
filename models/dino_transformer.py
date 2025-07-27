import torch
import torch.nn as nn
from transformers import AutoModel
class DinoTransformer(nn.Module):
    """PENet stripped down for classification with learned positional encoding for 24 slices."""

    def __init__(self, num_layers, num_slices=24, **kwargs):
        super(DinoTransformer, self).__init__()
        backbone = "dinov2_small"
        
        backbone_path = {
            'dinov2_small': 'facebook/dinov2-small',
            'dinov2_base': 'facebook/dinov2-base',
            'dinov2_large': 'facebook/dinov2-large',
            'dinov2_giant': 'facebook/dinov2-giant',
            'rad_dino': 'microsoft/rad-dino'
        }

        self.num_layers = num_layers
        self.embed_dim_dict = {'dinov2_small': 384, 'dinov2_base': 768, 'rad_dino': 768}
        self.backbone= AutoModel.from_pretrained(backbone_path[backbone])
        self.embed_dim = self.embed_dim_dict[backbone]

        # Learned positional embedding: (1, num_slices, embed_dim)
     #   self.positional_encoding = nn.Parameter(torch.randn(1, num_slices, self.embed_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Linear(self.embed_dim, 1)
        self.linear_frame = nn.Linear(self.embed_dim, 1)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x):
        # Expand grayscale to RGB
        x = x.expand(-1, 3, -1, -1, -1)  # (B,3,24,H,W)
        b, c, t, h, w = x.size()
        x = torch.reshape(x,(b * t, c, h, w))  # (B*T,3,H,W)
       # print(x.size())
        outputs = self.backbone(pixel_values=x)
      
        out = outputs.last_hidden_state[:, 0, :] 
       # print(out.size())
        out = out.view(b, t, self.embed_dim)  # (B, 24, embed_dim)
       # print(out.size())
        # Add positional encoding to per-slice features
     #   out = out + self.positional_encoding[:, :t, :]  # (B, 24, embed_dim)

        # Add CLS token
        
        cls_tokens = self.cls_emb.expand(b, -1, -1)  # (B,1,embed_dim)
       # print(self.cls_emb.size())
      #  print(cls_tokens.size())
        x = torch.cat((cls_tokens, out), dim=1)  # (B,25,embed_dim)
       # print(x.size())
        x = self.encoder(x)  # (B,25,embed_dim)

        x_cls = x[:, 0, :]  # (B, embed_dim)
        #x_frames = x[:, 1:, :]  # (B, 24, embed_dim)

        #output_frame = self.linear_frame(x_frames)  # (B,24,1)
        #output_frame = output_frame.permute(0, 2, 1)  # (B,1,24)

        output = self.linear(x_cls)  # (B,1)

        return output, x_cls


    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `PENet(**model_args)`.
        """
        model_args = {'num_layers' : self.num_layers}

        return model_args        

    def load_pretrained(self, ckpt_path, gpu_ids):
        """Load parameters from a pre-trained PENetClassifier from checkpoint at ckpt_path.
        Args:
            ckpt_path: Path to checkpoint for PENetClassifier.
        Adapted from:
            https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        """
        return

    def fine_tuning_parameters(self, fine_tuning_boundary, fine_tuning_lr=0.0):
        """Get parameters for fine-tuning the model.
        Args:
            fine_tuning_boundary: Name of first layer after the fine-tuning layers.
            fine_tuning_lr: Learning rate to apply to fine-tuning layers (all layers before `boundary_layer`).
        Returns:
            List of dicts that can be passed to an optimizer.
        """

        def gen_params(boundary_layer_name, fine_tuning):
            """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
            If unfrozen, generate the params at boundary_layer_name and beyond."""
            saw_boundary_layer = False
            for name, param in self.named_parameters():
                if name.startswith(boundary_layer_name):
                    saw_boundary_layer = True

                if saw_boundary_layer and fine_tuning:
                    return
                elif not saw_boundary_layer and not fine_tuning:
                    continue
                else:
                    yield param

        # Fine-tune the network's layers from encoder.2 onwards
        optimizer_parameters = [{'params': gen_params(fine_tuning_boundary, fine_tuning=True), 'lr': fine_tuning_lr},
                                {'params': gen_params(fine_tuning_boundary, fine_tuning=False)}]

        # Debugging info
        util.print_err('Number of fine-tuning layers: {}'
                       .format(sum(1 for _ in gen_params(fine_tuning_boundary, fine_tuning=True))))
        util.print_err('Number of regular layers: {}'
                       .format(sum(1 for _ in gen_params(fine_tuning_boundary, fine_tuning=False))))

        return optimizer_parameters
