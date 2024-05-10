import os
from torch.utils.data import ConcatDataset

from .dataset import AllWeather, LowLevel_Cityscapes, Raindrop, Snow100K, Synthetic_Rain, Outdoor_Rain, OTS
from .transforms_factory import create_transform
from configs import get_task_info, get_dataset_root
from utils import mprint

def create_dataset(augment_enable: bool = False, task_name: str = '', ds_name: str = '', 
                    split: str = 'train', random_crop_ratio: tuple = (1.0, 1.0), logger=None):

    if augment_enable:
        train = True if split == 'train' else False
        transforms = create_transform(task_name=task_name, ds_name=ds_name, train=train)
        mprint('data augmentation enable!!!')
    else:  # val and test
        transforms = None
        mprint('data augmentation disable!!!')

    dataset_root = get_dataset_root(dataset_name=ds_name)
    if 'allweather' in ds_name:
        dataset = AllWeather(dataset_root=dataset_root,
                             task='derain' if task_name == 'low_level' else task_name,
                             split=split,
                             transforms=transforms,
                             random_crop_ratio=random_crop_ratio)
    
    elif ds_name == 'cityscapes':
        if task_name == 'low_level' and split == 'train':  # low-level train
            task_list = get_task_info(ds_name, type='list')
            dataset_list = []
            for task in task_list:
                dataset = LowLevel_Cityscapes(dataset_root, task, split, transforms, random_crop_ratio)
                dataset_list.append(dataset)
            dataset = ConcatDataset(dataset_list)
        elif "infer" in split:
            task_list = get_task_info(ds_name, type='list')
            dataset_list = []
            for task in task_list:
                dataset = LowLevel_Cityscapes(dataset_root, task, split, transforms, random_crop_ratio)
                dataset_list.append(dataset)
            dataset = ConcatDataset(dataset_list)
        else:  # low_level val & test + specific training
            dataset = LowLevel_Cityscapes(dataset_root=dataset_root,
                                       task='derain' if task_name == 'low_level' else task_name,
                                       split='test',
                                       transforms=transforms,
                                       random_crop_ratio=random_crop_ratio)
    elif ds_name == 'raindrop':
        dataset = Raindrop(dataset_root, 'deraindrop', split, transforms, random_crop_ratio)
    elif ds_name == 'snow100k':
        dataset = Snow100K(dataset_root, 'desnow', split, transforms, random_crop_ratio)
    elif ds_name == 'synthetic_rain':
        dataset = Synthetic_Rain(dataset_root, 'derain', split, transforms, random_crop_ratio)
    elif ds_name == 'outdoor_rain':
        dataset = Outdoor_Rain(dataset_root, 'derain', split, transforms, random_crop_ratio)
    elif ds_name == 'ots':
        dataset = OTS(dataset_root, 'dehaze', split, transforms, random_crop_ratio)
    else:
        dataset = None
        raise NotImplementedError()

    if logger is not None:
        logger.info('task name: {}  dataset name: {}'.format(task_name, ds_name))
    else:
        mprint('task name: ', task_name)
        mprint('dataset name: ', ds_name)

    return dataset