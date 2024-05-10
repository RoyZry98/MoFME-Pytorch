import os
import sys

import csv
from collections import OrderedDict

def update_summary(epoch, metrics, filename, split, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    # rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    # rowd.update([('test_' + k, v) for k, v in test_metrics.items()])
    rowd.update([('{}_'.format(split) + k, v) for k, v in metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = round(self.sum / self.count, 5)
