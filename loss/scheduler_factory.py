import torch.optim.lr_scheduler as lr_scheduler
from .warmup_scheduler import GradualWarmupScheduler

from utils import mprint

def create_scheduler(scheduler_name: str, optimizer, epoch: int):
    
    mprint(scheduler_name)

    if scheduler_name == "multi_step":
        gamma = 0.5
        milestones = [100, 150]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_name == 'step':
        step_size = 25
        gamma = 0.5
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "cosine":
        T_max = epoch
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    elif scheduler_name == "cosine+warmup":
        T_max = epoch
        warmup_epochs = 3
        lr_min = 1e-6  # 100
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max-warmup_epochs, eta_min=lr_min)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    else:
        scheduler = None
        raise NotImplementedError()

    return scheduler