import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions.normal import Normal

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math
import copy
import time
import numpy as np

####################################  Top K  ###############################

class KeepTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: [b, n, d]
        if self.top_k == 0:
            return x
            
        filter_value=-float('Inf')
        indices_to_remove = x < torch.topk(x, self.top_k)[0][..., -1, None]  # topk返回value的最内层大小比较
        x[indices_to_remove] = filter_value

        return x