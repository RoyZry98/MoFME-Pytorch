import torch
import torch.backends.cudnn as cudnn

import os

import argparse
import numpy as np
import random
import logging

from utils._logging import setup_default_logging

# module
from data import create_dataset, create_loader
from configs import get_img_size, get_crop_ratio
from infering import create_infer_one_epoch

def main():
    """
    main()
    """

    # logger
    _logger = logging.getLogger('infer-MoWE')

    parser = argparse.ArgumentParser(description='MoWE Inference')

    ######### Basic Setting ##############

    parser.add_argument('--task', default='derain', type=str, help='task name')
    parser.add_argument('--dataset', default='maw_sim', type=str, help='dataset name')
    parser.add_argument('--split', default='infer+test', type=str,
                        choices=['infer+test', 'infer+train', 'infer+val'], help='dataset split')
    parser.add_argument('--model-path', default=None, type=str, help='inference model path')
    parser.add_argument('--model-name', default='mowe', type=str, help='inference model name')
    parser.add_argument('--exp', default='train', type=str, help='output dir')

    ######### GPU Setting ################

    parser.add_argument('--gpu-list', type=int, nargs='+', default=None, help='cuda list')

    ######### Inference Setting ################

    parser.add_argument('--bs', default=16, type=int, help='batch-size')
    parser.add_argument('--workers', default=15, type=int, help='num-workers')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    args = parser.parse_args()

    # device
    device = 'cuda' if len(args.gpu_list) > 0 else 'cpu'
    print('{}:'.format(device), args.gpu_list)

    gpus = ','.join([str(i) for i in args.gpu_list])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n")

    # output_dir
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    output_dir = os.getcwd() + '/output/infer/' + args.exp + '/' + args.task
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('==> Output dir:')
    print(output_dir)

    # logger
    setup_default_logging(default_level=logging.INFO, log_path="{}/output_info.log".format(output_dir))

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Data
    _logger.info('\n==> Preparing data..')
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
    _logger.info('\n==> Building model..')
    _logger.info('\n==> Load inference model from {}..'.format(args.model_path))
    model = torch.load(args.model_path, map_location=torch.device('cpu'))['net']

    model = model.to(device)
    if 'cuda' in device:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        cudnn.benchmark = True

    # Infering
    _logger.info('\n==> Infering..')

    infer_one_epoch = create_infer_one_epoch(model_name=args.model_name)

    infer_one_epoch(task_name=args.task,
                    dataset_name=args.dataset,
                    model=model,
                    model_name=args.model_name,
                    device=device,
                    testloader=testloader,
                    crop_ratio=crop_ratio,
                    output_dir=output_dir,
                    logger=_logger,)

if __name__ == "__main__":
    main()