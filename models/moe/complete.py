import math
import copy
import time

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from utils import to_2tuple

from ..mowe_basic_module import PreNorm, Attention, FeedForward, PrintLayer, PatchEmbed, FFN_Detail, \
                                FeedForward_SA
from .basic_module import KeepTopK


class Transformer_MoE(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., num_expert=0,
                 top_k=0, type_expert='ffn', gate='complete', is_single_task=False, *args, **kwargs):
        super().__init__()

        self.num_expert = num_expert

        TopK_Function = KeepTopK(top_k=int(top_k))
        router = nn.Sequential(
                    nn.Linear(dim, num_expert),
                    TopK_Function,
                    nn.Softmax(dim=-1)
                )

        self.gate = gate

        self.is_single_task = is_single_task

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # [attention, router, [ffn_1, ffn_2, ...]]
            attn = PreNorm(dim, Attention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout))
            module_list = [attn, copy.deepcopy(router)]
            if type_expert == 'detail_1357':
                kernel_list = [0, 3, 5, 7]
                module_list += [nn.ModuleList([
                    PreNorm(dim, FFN_Detail(dim, mlp_dim, dim, kernel_size=kernel_list[i%4])) for i in range(num_expert)
                ])]
            elif type_expert == 'detail_3333':
                kernel_list = [3, 3, 3, 3]
                module_list += [nn.ModuleList([
                    PreNorm(dim, FFN_Detail(dim, mlp_dim, dim, kernel_size=kernel_list[i%4])) for i in range(num_expert)
                ])]
            elif type_expert == 'sa':
                module_list += [nn.ModuleList([
                    PreNorm(dim, FeedForward_SA(dim, mlp_dim, dropout)) for i in range(num_expert)
                ])]
            else:
                module_list += [nn.ModuleList([PreNorm(dim, FeedForward(dim, mlp_dim, dropout)) for _ in range(num_expert)])]
            self.layers.append(nn.ModuleList(module_list))

    def forward(self, x, *args, **kwargs):

        weights_list = []
        B, N, D = x.shape
        for i, (attn, router, ff_list) in enumerate(self.layers):
            
            x = attn(x) + x

            weights = router(x)  #token level routing, [b, n, d]

            y = 0
            for idx in range(self.num_expert):
                weight_idx = weights[:, :, idx].unsqueeze(dim=-1)
                y += weight_idx * ff_list[idx](x)

            x = y + x  # x: residual, y: experts-ffn

            weights_list.append(weights)

        return x, weights_list


class ViT_MoE(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, n_heads, mlp_dim, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., num_token=0, num_expert=0, top_k=0,
                 type_expert='ffn', gate='complete', is_single_task=False, *args, **kwargs):

        super().__init__()

        self.num_expert = num_expert

        img_height, img_width = to_2tuple(img_size)
        patch_height, patch_width = to_2tuple(patch_size)

        assert img_height % patch_height == 0 and img_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        self.to_patch_embedding = PatchEmbed(img_size=to_2tuple(img_size), patch_size=to_2tuple(patch_size),
                                            in_channels=channels, embed_dim=embed_dim)
        self.img_pos_embedding = nn.Parameter(torch.randn(1, num_token, embed_dim))

        self.gate = gate
        self.transformer = Transformer_MoE(embed_dim, depth, n_heads, dim_head, mlp_dim, dropout,
                                            num_expert=num_expert, type_expert=type_expert, top_k=top_k, 
                                            gate=gate, is_single_task=is_single_task)

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img, task_idx: int):

        img_tokens = self.to_patch_embedding(img)

        x = img_tokens
        # x += self.img_pos_embedding

        x = self.dropout(x)

        x, weights_list = self.transformer(x, task_idx)
        
        l_aux = None
        cls_output = None
            
        return x, cls_output, weights_list, l_aux