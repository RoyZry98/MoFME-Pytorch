from configs import _model_register_list
from utils import mprint


def create_train_one_epoch(model_name: str):
    if model_name in ['mowe']:
        from .train_lowlevel_mowe import train_one_epoch
        mprint('from train_lowlevel_mowe')
                
    else:
        raise NotImplementedError


    return train_one_epoch