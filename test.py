import torch
import torch.backends.cudnn as cudnn
import os

import argparse
import numpy as np
import random
import logging

from utils._logging import setup_default_logging

# module
from configs import get_img_size, get_crop_ratio
from data import create_dataset, create_loader
from testing import create_test_one_epoch
from utils import update_summary, setup_default_logging
from loss import create_loss


def main():
    """
    main()
    """

    # logger
    _logger = logging.getLogger('test-MoWE')

    parser = argparse.ArgumentParser(description='MoWE Test')

    ######### Basic Setting ##############

    parser.add_argument('--task', default='derain', type=str, help='task name')
    parser.add_argument('--dataset', default='maw_sim', type=str, help='dataset name')
    parser.add_argument('--split', default='test', type=str,
                        choices=['test', 'test+a', 'test+b', 'test+s', 'test+m', 'test+l',
                                 'test+100h', 'test+100l', 'test+100', 'test+1200', 'test+2800'], 
                        help='dataset split')
    parser.add_argument('--model-path', default=None, type=str, help='inference model path')
    parser.add_argument('--model-name', default='mowe', type=str, help='inference model name')
    parser.add_argument('--loss-list', type=str, nargs='+', default=None, metavar='LOSS-LIST',
                        help='Loss list (format: [function, specific_name, weight, ...])')
    parser.add_argument('--exp', default='test', type=str, help='output dir')

    ######### GPU Setting ################

    parser.add_argument('--gpu-list', type=int, nargs='+', default=None, help='cuda list')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')

    ######### Testing Setting ################

    parser.add_argument('--bs', default=16, type=int, help='batch-size')
    parser.add_argument('--overlap-crop', default=False, action='store_true', help='enable overlap crop')
    parser.add_argument('--workers', default=15, type=int, help='num-workers')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    ############## Common Print Setting ###############
    parser.add_argument('--gate-print', default=False, action='store_true', help='print the weights of moe gate ')

    args = parser.parse_args()

    # DP
    device = 'cuda' if len(args.gpu_list) > 0 else 'cpu'
    print('{}:'.format(device), args.gpu_list)

    assert torch.cuda.is_available(), "Testing currently requires at least one GPU."
    gpus = ','.join([str(i) for i in args.gpu_list])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n")

    # random seed
    seed = args.seed
    random.seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # output_dir
    output_dir = os.getcwd() + '/output/test/' + args.exp + '/' + args.task
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('==> Output dir:')
    print(output_dir)

    # logger
    setup_default_logging(default_level=logging.INFO, log_path="{}/output_info.log".format(output_dir))

    # Data
    print('\n==> Preparing data..')

    img_size = get_img_size(dataset_name=args.dataset)
    crop_ratio = get_crop_ratio(dataset_name=args.dataset)

    dataset = create_dataset(task_name=args.task,
                             ds_name=args.dataset,
                             split=args.split,
                             random_crop_ratio=(1.0, 1.0))
    
    testloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=args.bs, 
                        shuffle=False, 
                        num_workers=args.workers, 
                        pin_memory=True
                    )

    # Model
    print('\n==> Building model..')
    print('Load inference model from {}..'.format(args.model_path))
    ckpt = torch.load(args.model_path, map_location=torch.device('cpu'))
    model = ckpt['net']
    # print(model)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    print("Best Epoch: {}, PSNR: {}, SSIM: {}".format(ckpt['epoch'], ckpt['psnr'], ckpt['ssim']))

    model = model.to(device)
    if 'cuda' in device:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        cudnn.benchmark = True
        
    # Loss
    print('\n==> Creating loss..')
    loss_dict = create_loss(task_name=args.task, loss_list=args.loss_list)

    # Testing
    print('\n==> Testing for task []..'.format(args.task))

    test_one_epoch = create_test_one_epoch(model_name=args.model_name, train=False)

    test_one_epoch(task_name=args.task,
                   is_downstream_finetune=False,
                   epoch=1,
                   total_epoch=1,
                   model_name=args.model_name,
                   model=model,
                   device=device,
                   testloader=testloader,
                   loss_dict=loss_dict,
                   best_metrics=(0, 0, 0),
                   crop_ratio=crop_ratio,
                   overlap_crop=args.overlap_crop,
                   output_dir=output_dir,
                   gate_print=args.gate_print,
                   logger=_logger,
                   wandb_enable=False)

if __name__ == "__main__":
    main()