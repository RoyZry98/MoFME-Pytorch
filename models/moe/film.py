import copy

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from utils import to_2tuple

from ..mowe_basic_module import PreNorm, Attention, FeedForward, PrintLayer, PatchEmbed
from .basic_module import KeepTopK


####################################  FiLM Modulation MoE  ###############################

class Modular_Experts(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expert = nn.ModuleList([
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
            ])
        self.norm = nn.LayerNorm(dim)
        # self.act = nn.Identity()

    def forward(self, x, *args, **kwargs):
        # x: [b, n, d]
        conv_product, conv_add = self.expert
        modulation_product, modulation_add = conv_product(x), conv_add(x)
        x = modulation_product * x + modulation_add
        return x

class Transformer_MoE_FiLM(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., num_expert=0,
                 top_k=0, gate='film', is_single_task=False, *args, **kwargs):
        super().__init__()

        self.num_expert = num_expert

        TopK_Function = KeepTopK(top_k=int(top_k))

        router = nn.Sequential(
                    nn.Linear(dim, num_expert),
                    TopK_Function,
                    nn.Softmax(dim=-1)
                )

        adaptor = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        self.gate = gate

        self.is_single_task = is_single_task

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # [attention, router, adaptor, [ffn_1, ffn_2, ...]]
            attn = PreNorm(dim, Attention(dim, num_heads=heads, dim_head=dim_head, dropout=dropout))
            module_list = [attn, copy.deepcopy(router), copy.deepcopy(adaptor)]

            conv2d = nn.Linear(dim, dim)
            basenet = nn.ModuleList([
                    nn.Sequential(copy.deepcopy(conv2d)),
                    nn.Sequential(copy.deepcopy(conv2d)),
                ])
            act = nn.GELU()
            module_list += [nn.ModuleList([nn.LayerNorm(dim), basenet, act]+[Modular_Experts(dim) for _ in range(num_expert)])]

            self.layers.append(nn.ModuleList(module_list))

    def forward(self, x, *args, **kwargs):

        weights_list = []
        bs, n, d = x.shape
        device = x.device
        estimate_time = 5
        uncertainty_list = []

        for i, (attn, router, adaptor, ff_list) in enumerate(self.layers):
            
            x = attn(x) + x

            # uncertainty estimation
            if self.training:
            # Compute uncertainty only during training
                for i in range(estimate_time):
                    weights = router(x)  # origin: token level routing, [b, n, d]
                    uncertainty_list.append(weights)
                weights = torch.mean(torch.stack(uncertainty_list, dim=0), dim=0)  # Average over all samples
                
                # Calculate covariance matrix
                weights_centered = torch.stack(uncertainty_list, dim=0) - weights.unsqueeze(0)  # Center the weights [estimate_time, b, n, num_expert]
                cov = torch.einsum('ijkl,ijkm->jklm', weights_centered, weights_centered) / (estimate_time - 1)  # [b, n, num_expert, num_expert]
                cov_inv = torch.inverse(cov + torch.eye(cov.shape[-1], device=device).unsqueeze(0).unsqueeze(0) * 1e-5)  # Add regularization for numerical stability
    
                # Normalize each sample
                normalized_weights_list = []
                for weights_sample in uncertainty_list:
                    diff = weights_sample - weights  # [b, n, num_expert]
                    transformed_diff = torch.einsum('ijk,jklm->ijlm', diff, cov_inv)  # [b, n, num_expert]
                    norm = torch.norm(transformed_diff, dim=2, keepdim=True)  # [b, n, 1]
                    normalized_weights = transformed_diff / (norm + 1e-5)  # Avoid division by zero
                    normalized_weights_list.append(normalized_weights)
    
                # Optionally, use the normalized weights directly or some function of them
                weights = torch.mean(torch.stack(normalized_weights_list, dim=0), dim=0)  # Average normalized weights
            else:
                weights = router(x)  # origin: token level routing, [b, n, d]
                
            y = 0

            # basenet
            layernorm, basenet, act = ff_list[0], ff_list[1], ff_list[2]
            y_temp = layernorm(x)
            
            y_temp = act(basenet[0](y_temp)) * basenet[1](y_temp)
            
            # film experts
            for idx in range(self.num_expert):
                weight_idx = weights[:, :, idx].unsqueeze(dim=-1)
                y += weight_idx * ff_list[idx+2](y_temp)
            y = adaptor(y)

            x = y + x  # x: residual, y: experts-ffn

            weights_list.append(weights)

        return x, weights_list


class ViT_MoE_FiLM(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, n_heads, mlp_dim, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., num_token=0, num_expert=0, top_k=0,
                 gate='film', is_single_task=False, *args, **kwargs):

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
        self.transformer = Transformer_MoE_FiLM(embed_dim, depth, n_heads, dim_head, mlp_dim, dropout,
                                                num_expert=num_expert, top_k=top_k, gate=gate, 
                                                is_single_task=is_single_task)

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
