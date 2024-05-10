
from .transforms import data_augment_lowlevel_torch

def create_transform(task_name: str = '', ds_name: str = '', train: bool = True):

    if train:
        transforms = data_augment_lowlevel_torch
    else:
        transforms = None

    return transforms
