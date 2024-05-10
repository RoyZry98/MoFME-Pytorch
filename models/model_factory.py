import torch
from .mowe import MoWE
from utils import mprint


def create_model(task_name: str,
                 task_scale: str,
                 img_size: tuple,
                 patch_size: int,

                 model_name: str,
                 residual_learning: bool,
                 embed_dim: int,
                 scale_dim: int,
                 depth: int,
                 transformer_type: str = 'vit',  # image transformer

                 moe_enable: bool = False,
                 num_expert: int = 4,
                 type_expert: str = 'ffn',
                 top_k: int = 0,
                 gate: str = 'complete',  # image moe transformer gate mechanism
                 finetune_ratio: float = None,

                 resume: str = '',
                 logger=None):

    if logger is not None:
        logger.info(model_name)
    else:
        mprint(model_name)

    if model_name == 'mowe':
        model = MoWE(task_scale=task_scale,
                    is_single_task=(task_name != 'low_level'),

                    img_size=img_size,
                    patch_size=patch_size,

                    residual_learning=residual_learning,
                    transformer_type=transformer_type,
                    embed_dim=embed_dim,
                    scale_dim=scale_dim,
                    depth=depth,

                    moe_enable=moe_enable,
                    num_expert=num_expert,
                    type_expert=type_expert,
                    top_k=top_k,
                    gate=gate,)
    else:
        raise NotImplementedError
    # print(model)

    mprint('Model[{}] has been set for task:[{}]'.format(model_name, task_name))

    if resume != '':
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model