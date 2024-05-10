from configs import _model_register_list
from utils import mprint

def create_test_one_epoch(model_name: str, train: bool = True):

    if model_name == 'mowe':
        if train:
            from .test_single_mowe import test_one_epoch
            mprint('from test_single_mowe')
        else:
            from .test_single_mowe_dp import test_one_epoch
            print('from test_single_mowe_dp')
    
    else:
        raise NotImplementedError

    return test_one_epoch