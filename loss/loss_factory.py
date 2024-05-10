from torch import nn

from .perception_loss import Perception_Loss
from utils import nested_dict, mprint

def create_loss(task_name: str, loss_list: str):

    '''
    loss_factory[task][function][specific_name]
    loss_list: [function, specific_name, weight, ...]
    loss_dict:
    '''
    loss_factory = nested_dict()

    loss_factory['low_level']['content']['l1'] = nn.SmoothL1Loss()
    loss_factory['low_level']['perception']['vgg16'] = Perception_Loss()

    loss_factory['derain']['content']['l1'] = nn.SmoothL1Loss()
    loss_factory['derain']['perception']['vgg16'] = Perception_Loss()

    loss_factory['deraindrop']['content']['l1'] = nn.SmoothL1Loss()
    loss_factory['deraindrop']['perception']['vgg16'] = Perception_Loss()

    loss_factory['dehaze']['content']['l1'] = nn.SmoothL1Loss()
    loss_factory['dehaze']['perception']['vgg16'] = Perception_Loss()

    loss_factory['desnow']['content']['l1'] = nn.SmoothL1Loss()
    loss_factory['desnow']['perception']['vgg16'] = Perception_Loss()

    loss_dict = nested_dict()  # {}
    total_weight = 0
    loss_list_len = len(loss_list)
    assert loss_list_len % 3 == 0, "loss_list_length must be divided by 3!!!"

    for idx in range(loss_list_len//3):
        _function, _name, _weight = loss_list[3*idx], loss_list[3*idx+1], float(loss_list[3*idx+2])
        total_weight += _weight
        loss_dict[task_name][_function][_name]['loss'] = loss_factory[task_name][_function][_name]
        loss_dict[task_name][_function][_name]['weight'] = _weight
        mprint('Loss: {} {} {} {}'.format(task_name, _function, _name, _weight))

    # assert total_weight == 1, "loss total weight must be 1, not {}".format(total_weight)

    return loss_dict