import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import copy
import random

from utils import mprint

def create_loader(dataset: Dataset, ds_name: str, task_name: str, rank: int, global_seed: int, 
                  batch_size: int, shuffle: bool = False, num_workers: int = 0, ):
    
    mprint('[{}] dataset_length: [{}]'.format(ds_name, len(dataset)))
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=global_seed
    )    
    loader = DataLoader(
            dataset, 
            batch_size=int(batch_size // dist.get_world_size()), 
            shuffle=False, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=True)

    return loader