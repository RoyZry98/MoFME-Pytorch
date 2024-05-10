import copy
import torch
from torch import nn
import torch.nn.functional as F

from utils import to_2tuple

from ..mowe_basic_module import PreNorm, Attention, FeedForward, PrintLayer, PatchEmbed

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, num_heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, *args, **kwargs):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):  # baseline
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., *args, **kwargs):

        super().__init__()

        image_height, image_width = to_2tuple(image_size)
        patch_height, patch_width = to_2tuple(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = PatchEmbed(img_size=to_2tuple(image_size), patch_size=to_2tuple(patch_size),
                                            in_channels=channels, embed_dim=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

    def forward(self, img, *args, **kwargs):
        x = self.to_patch_embedding(img)

        # x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = self.to_latent(x)

        return x, None, None, None  # x, cls_output, weights_list, l_aux