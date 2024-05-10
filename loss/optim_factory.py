import torch.optim as optim

from utils import mprint

def create_optimizer(optimizer_name: str, model, lr):

    mprint(optimizer_name)

    if optimizer_name == "sgd":
        momentum = 0.9
        weight_decay = 5e-4
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
                              
    elif optimizer_name == "adam":
        betas = (0.9, 0.999)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    elif optimizer_name == "adamw":
        betas = (0.9, 0.999)
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas)

    return optimizer