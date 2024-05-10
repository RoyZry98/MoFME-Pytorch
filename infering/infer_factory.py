from configs import _model_register_list
from utils import mprint

def create_infer_one_epoch(model_name: str):

    if model_name == 'mowe':
        from .infer_single_mowe import infer_one_epoch
        mprint('from infer_single_mowe')

    else:
        raise NotImplementedError

    return infer_one_epoch