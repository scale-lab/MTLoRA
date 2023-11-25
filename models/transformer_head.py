import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer, PatchMerging, PatchEmbed


class UpSample(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        return x


class SwinDecoderHead(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override downsample layers with upsampling
        self.downsample = nn.ModuleList([
            UpSample(embed_dim=kwargs['embed_dim'] * 2**i) for i in range(self.num_layers)
        ])

    def forward_features(self, x, return_stages=False, flatten_ft=True):
        return_stages = False
        flatten_ft = True
        
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        if return_stages:
            out = []
        for layer in self.layers:
            x = layer(x)
            if return_stages:
                out.append(x)
        return x

# ([4, 3, 224, 224])
# ([4, 270, 28, 28])
