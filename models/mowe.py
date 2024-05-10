
import torch
from torch import nn
from einops.layers.torch import Rearrange

from utils import to_2tuple, mprint

from .mowe_basic_module import default_conv, ResBlock, Upsampler, PrintLayer
from .moe import ViT, get_vit_moe

class Image_Transformer_Encoder(nn.Module):
    def __init__(self,
                 is_single_task,

                 img_size,
                 in_ch,
                 patch_size,

                 embed_dim,
                 depth,
                 n_heads,
                 mlp_dim,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 num_token=0,

                 transformer_type='vit',
                 num_expert=0,
                 type_expert='ffn',
                 num_task=0,
                 top_k=0,
                 moe_enable=False,
                 gate='baseline',
                 ):

        super().__init__()

        self.transformer_type = transformer_type
        if transformer_type == 'vit' and moe_enable:
            self.transformer_type = 'vit_moe'

        if self.transformer_type == 'vit_moe':
            vit = get_vit_moe(gate)
            self.vision_transformer = vit(
                img_size=img_size,
                channels=in_ch,
                patch_size=patch_size,

                embed_dim=embed_dim,
                depth=depth,
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                dropout=dropout,
                emb_dropout=emb_dropout,
                num_token=num_token,

                num_expert=num_expert,
                type_expert=type_expert,
                num_task=num_task,
                top_k=top_k,
                gate=gate,
                is_single_task=is_single_task,
            )

        elif self.transformer_type == 'vit':  # baseline
            self.vision_transformer = ViT(
                image_size=img_size, 
                channels=in_ch,
                patch_size=patch_size, 

                dim=embed_dim, 
                depth=depth, 
                heads=n_heads, 
                mlp_dim=mlp_dim, 
                dim_head=dim_head, 
                dropout=dropout, 
                emb_dropout=emb_dropout
            )

        else:
            raise NotImplementedError()

    def forward(self, img, task_idx: int):
        
        x, cls_output, weights_list, l_aux = self.vision_transformer(img, task_idx)

        return x, cls_output, weights_list, l_aux


class MoWE(nn.Module):
    def __init__(self,
                 task_scale: str = '1+1+1',  # derain, dehaze, desnow
                 is_single_task: bool = False,

                 img_size=512,  # int or tuple
                 in_ch: int = 3,
                 out_ch: int = 32,
                 patch_size: int = 8,
                 
                 residual_learning: bool = True,
                 transformer_type: str = 'vit',

                 moe_enable: bool = False,
                 num_expert: int = 4,
                 type_expert: str = 'ffn',
                 top_k: int = 0,
                 gate: str = 'baseline',

                 embed_dim: int = 256,  # embed_dim = dim_head * n_heads !!!
                 dim_head: int = 64,
                 n_heads: int = 8,  # origin: 4
                 depth: int = 2,  # origin: 4
                 scale_dim: float = 4,  # mlp_dim = dim * scale_dim
                 dropout: float = 0.1,
                 embed_dropout: float = 0,

                 kernel_size: int = 3,
                 
                 out_ch_scale: int = 4
                 ):
        super().__init__()

        assert isinstance(img_size, tuple) or isinstance(img_size, list) or isinstance(img_size, int), \
            "img_size must be tuple, list or int, not {}".format(type(img_size))

        task_scale = list(map(lambda x: int(x), task_scale.split('+')))  # example: '1+1+1' --> [1, 1, 1]
        mprint('task scale:', task_scale)
        num_task = len(task_scale)
        mprint('num task:', num_task)

        self.residual_learning = residual_learning  # image transformer residual learning
        if not self.residual_learning:
            mprint("disable image residual learning !!!")

        img_size = to_2tuple(img_size)  # unified tuple format

        patch_dim = out_ch * patch_size ** 2
        _h = img_size[0] // patch_size
        _w = img_size[1] // patch_size

        self.patch_embed_to_img_feature = nn.Sequential(
                nn.Linear(embed_dim, patch_dim * out_ch_scale),
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=_h)
            ) 

        img_h = img_size[0]
        img_w = img_size[1]
        assert img_h % patch_size == 0 and img_w % patch_size == 0, \
            'Image dimensions must be divisible by the patch size.'
        num_token = (img_h // patch_size) * (img_w // patch_size)
        self.num_token = num_token

        self.moe_enable = moe_enable

        ''' image encoder'''
        self.Img_Encoder = Image_Transformer_Encoder(
            is_single_task=is_single_task,

            img_size=img_size,
            in_ch=out_ch,
            patch_size=patch_size,

            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=int(embed_dim * scale_dim),
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=embed_dropout,
            num_token=num_token,

            transformer_type=transformer_type,
            num_expert=num_expert,
            type_expert=type_expert,
            num_task=num_task,
            top_k=top_k,
            moe_enable=moe_enable,
            gate=gate,
        )

        conv = default_conv
        self.head = nn.Sequential(
                        conv(in_ch, out_ch, kernel_size),
                        ResBlock(conv, out_ch, kernel_size),
                        ResBlock(conv, out_ch, kernel_size)
                    )

        ch = out_ch * out_ch_scale
        self.tail = nn.Sequential(
                        Upsampler(conv, 1, ch),  # nn.pixelshuffle upsampler
                        ResBlock(conv, ch, kernel_size),
                        conv(ch, ch // 2, kernel_size),
                        ResBlock(conv, ch // 2, kernel_size),
                        conv(ch // 2, in_ch, kernel_size),
                        nn.Sigmoid()
                    )

    def forward(self, img_lq, task_idx=None):

        if task_idx is None:
            task_idx = torch.ones(img_lq.shape[0], dtype=torch.long, device=img_lq.device)  # [bs]

        img_lq_feature = self.head(img_lq)  # img_lq_feature: [bs, out_ch, h, w]

        dec_output, cls_output, weights_list, l_aux = self.Img_Encoder(img_lq_feature, task_idx)

        img_lq_feature = dec_output

        img_hq_feature = self.patch_embed_to_img_feature(img_lq_feature)  # adjust dimension & reshape
        img_hq = self.tail(img_hq_feature)

        if self.residual_learning:
            img_hq = img_hq + img_lq

        return img_hq, cls_output, weights_list, l_aux