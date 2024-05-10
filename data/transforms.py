import torch

import random

def random_crop(lq, gt, random_crop_ratio):
    # lq: low quality image, type: Tensor
    h, w = lq.shape[1], lq.shape[2]
    crop_h, crop_w = int(h * random_crop_ratio[0]), int(w * random_crop_ratio[1])
    assert crop_h == crop_w, "Image crop area should be a square, but got ({}, {})".format(crop_h, crop_w)
    crop = crop_h

    rr = random.randint(0, h - crop)
    cc = random.randint(0, w - crop)

    lq = lq[:, rr:rr+crop, cc:cc+crop]
    gt = gt[:, rr:rr+crop, cc:cc+crop]

    return lq, gt

def data_augment_lowlevel_torch(input, mode):
    '''
    input: image [3, h, w] may be input or target
    mode: 0, 1, 2, 3, ..., 7 totall 8
    '''

    # Data Augmentations
    if mode == 0:
        output = input
    elif mode == 1:
        output = input.flip(1)
    elif mode == 2:
        output = input.flip(2)
    elif mode == 3:
        output = torch.rot90(input, dims=(1, 2))
    elif mode == 4:
        output = torch.rot90(input, dims=(1, 2), k=2)
    elif mode == 5:
        output = torch.rot90(input, dims=(1, 2), k=3)
    elif mode == 6:
        output = torch.rot90(input.flip(1), dims=(1, 2))
    elif mode == 7:
        output = torch.rot90(input.flip(2), dims=(1, 2))
    else:
        raise NotImplementedError()

    return output