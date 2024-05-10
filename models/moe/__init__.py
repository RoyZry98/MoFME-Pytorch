from .baseline import ViT
from .complete import ViT_MoE
from .film import ViT_MoE_FiLM
from utils import mprint

def get_vit_moe(gate: str):
    
    mprint('gate: {}'.format(gate))

    vit_dict = {
        'complete': ViT_MoE,
        'film': ViT_MoE_FiLM,
    }

    return vit_dict[gate]