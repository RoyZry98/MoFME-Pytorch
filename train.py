import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time
import argparse
import numpy as np
import random
import logging
import copy

# module
from configs import get_img_size, get_crop_ratio, get_task_info, get_no_val_dataset
from data import create_dataset, create_loader
from loss import create_loss, create_optimizer, create_scheduler
from models import create_model
from metrics import create_metrics_initial
from utils import update_summary, setup_default_logging, mprint, cleanup
from training import create_train_one_epoch
from testing import create_test_one_epoch

import wandb
import warnings
warnings.filterwarnings("ignore") 

from ptflops import get_model_complexity_info

def main():
    """
    main()
    """

    # logger
    _logger = logging.getLogger('train-MoWE')

    parser = argparse.ArgumentParser(description='MoWE Training')

    ######### Basic Setting ##############

    parser.add_argument('--task', default='deblur', type=str, help='task name')
    parser.add_argument('--dataset', default='maw_sim', type=str, help='dataset name')
    parser.add_argument('--model', default='mowe', type=str, help='model name')
    parser.add_argument('--pretrain', action='store_true', help='load pretrain model')
    parser.add_argument('--exp', default='train', type=str, help='output dir')

    ######### GPU Setting ################

    parser.add_argument('--gpu-list', type=int, nargs='+', default=None, help='cuda list')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')

    ######### Training Setting ################

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epoch')
    parser.add_argument('--global-bs', default=16, type=int, help='batch-size')
    parser.add_argument('--loss-list', type=str, nargs='+', default=None, metavar='LOSS-LIST',
                        help='Loss list (format: [function, specific_name, weight, ...])')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')
    parser.add_argument('--scheduler', default='multi_step', type=str, help='scheduler')
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--workers', default=15, type=int, help='num-workers')
    parser.add_argument('--global-seed', default=42, type=int, help='random global seed')

    parser.add_argument('--test-interval', default=1, type=int, help='test epoch interval for training, default 1')
    parser.add_argument('--overlap-crop', default=False, action='store_true', help='enable overlap crop')

    ######### Image Setting #####################

    parser.add_argument('--img-patch', default=4, type=int, help='img patch size')
    parser.add_argument('--augment-enable', default=False, action='store_true', help='enabled data augmentation')
    
    parser.add_argument('--embed-dim', default=256, type=int, help='embedding dimension')
    parser.add_argument('--scale-dim', default=4, type=int, help='mlp scale embedding dimension ratio, default: 4')
    parser.add_argument('--layer', default=2, type=int, help='Transformer layers')
    parser.add_argument('--disable-residual', action='store_true', help='disable residual learning, default:False')

    ############## MoE Setting ##################

    parser.add_argument('--moe-enable', default=False, action='store_true', help='enabled moe')
    parser.add_argument('--num-expert', default=4, type=int, help='num of experts')
    parser.add_argument('--type-expert', default='ffn', type=str, 
                                    choices=['ffn', 'detail_1357', 'detail_3333', 'sa'],
                                    help='type of expert')
    parser.add_argument('--top-k', default=0, type=int, help='each token choose top k experts, default: 0 --> all')
    parser.add_argument('--gate', default='baseline', type=str, 
                                choices=['complete', 'film'],
                                help='gate mechanism')

    ############## Common Print Setting ###############

    parser.add_argument('--gate-print', default=False, action='store_true', help='print the weights of moe gate ')

    ############## Summary Setting ###############

    parser.add_argument('--wandb-enable', action='store_true', help='enabled wandb')
    parser.add_argument('--wandb-project', default="spikefusion", type=str,
                        help='wandb project name, only used when wandb is enabled')
    parser.add_argument('--wandb-entity', default="lyl010221", type=str,
                        help='wandb entity name, only used when wandb is enabled')

    args = parser.parse_args()

    # DP
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    gpus = ','.join([str(i) for i in args.gpu_list])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_bs % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    
    # random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # output_dir
    output_dir = os.getcwd() + '/output/train/' + args.exp
    if rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, 0o777)
        mprint('==> Output dir:')
        mprint(output_dir)

        # wandb
        if args.wandb_enable:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity)
            wandb.run.name = args.exp

        # logger
        setup_default_logging(default_level=logging.INFO, log_path="{}/output_info.log".format(output_dir))


    # Data
    mprint('\n==> Preparing data..')
    
    img_size = get_img_size(dataset_name=args.dataset)
    crop_ratio = get_crop_ratio(dataset_name=args.dataset)
    
    trainset = create_dataset(augment_enable=args.augment_enable,  # default False
                              task_name=args.task,
                              ds_name=args.dataset,
                              split='train',
                              random_crop_ratio=crop_ratio)
    trainloader = create_loader(dataset=trainset, ds_name=args.dataset, 
                                task_name=args.task, batch_size=args.global_bs,
                                rank=rank, global_seed=args.global_seed,
                                shuffle=True, num_workers=args.workers)

    if args.dataset not in get_no_val_dataset():
        valset = create_dataset(task_name=args.task,
                                ds_name=args.dataset,
                                split='val',
                                random_crop_ratio=(1.0, 1.0))
        valloader = create_loader(valset, ds_name=args.dataset, task_name=args.task,
                                batch_size=dist.get_world_size(),
                                 rank=rank, global_seed=args.global_seed,
                                  shuffle=False, num_workers=args.workers)

    testset = create_dataset(task_name=args.task,
                             ds_name=args.dataset,
                             split='test',
                             random_crop_ratio=(1.0, 1.0))

    testloader = create_loader(testset, ds_name=args.dataset, 
                               task_name=args.task, batch_size=dist.get_world_size(),
                               rank=rank, global_seed=args.global_seed,
                               shuffle=False, num_workers=args.workers)

    # Model
    mprint('\n==> Building model..')
    img_size = (int(img_size[0] * crop_ratio[0]), int(img_size[1] * crop_ratio[1]))

    mprint('\n==> Training from scratch..')
    model = create_model(task_name=args.task,
                        task_scale=get_task_info(dataset=args.dataset, type='scale'),

                        model_name=args.model,
                        residual_learning=(not args.disable_residual),

                        gate=args.gate,
                        moe_enable=args.moe_enable,
                        num_expert=args.num_expert,
                        type_expert=args.type_expert,
                        top_k=args.top_k,

                        img_size=img_size,
                        patch_size=args.img_patch,
                        embed_dim=args.embed_dim,
                        scale_dim=args.scale_dim,
                        depth=args.layer,

                        resume=args.resume,
                        logger=_logger)
    
    if rank == 0:  # param & infer time
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        h, w = img_size[0], img_size[1]
        flops, params = get_model_complexity_info(model_copy, (3, h, w), as_strings=True, print_per_layer_stat=True)
        
        mprint('Flops:  ' + flops)
        mprint('Params: ' + params)
        
        t = 0
        img = torch.randn(size=(1, 3, h, w)).cuda(0)
        model_copy = model_copy.cuda(0)
        for i in range(220):
            if i >= 20:
                t1 = time.time()
            output = model_copy(img)
            if i >= 20:
                t2 = time.time()
                t += (t2-t1)
        mprint('Times per image: {}'.format(t/200))


    # DDP
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    # Loss
    mprint('\n==> Creating loss..')
    loss_dict = create_loss(task_name=args.task, loss_list=args.loss_list)

    # Optimizer
    # bs = 32 -> lr = base_lr
    mprint('\n==> Creating optimizer .. [lr={}x{}]'.format(args.lr, args.global_bs/32.0))
    args.lr *= args.global_bs/32.0
    optimizer = create_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)

    # Scheduler
    mprint('\n==> Creating scheduler..')
    scheduler = create_scheduler(scheduler_name=args.scheduler, optimizer=optimizer, epoch=args.epoch)

    if args.wandb_enable and rank == 0:
        wandb.config = {
            "model_name": args.model,
            "learning_rate": args.lr,
            "epochs": args.epoch,
            "batch_size": args.global_bs,
        }

    # Best Metrics Initialization
    mprint('\n==> Create metrics initial..')
    if args.dataset in ['allweather', 'cityscapes'] and args.task == 'low_level':
        task_name = 'derain'
    else:
        task_name = args.task
    best_metrics = create_metrics_initial(task_name=task_name)
    mprint(best_metrics)

    # Training & Testing
    mprint('\n==> Training..')

    train_one_epoch = create_train_one_epoch(model_name=args.model)
    test_one_epoch = create_test_one_epoch(model_name=args.model)
    
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch
    for epoch in range(start_epoch, start_epoch + args.epoch):
        if epoch == 1:
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()

        if args.wandb_enable and rank == 0:
            wandb.watch(model, log="all")

        train_metrics = train_one_epoch(task_name=args.task,
                                        epoch=epoch,
                                        total_epoch=args.epoch,
                                        model=model,
                                        model_name=args.model,
                                        moe_enable=args.moe_enable,
                                        gate=args.gate,
                                        device=device,
                                        trainloader=trainloader,
                                        optimizer=optimizer,
                                        loss_dict=loss_dict,
                                        gate_print=args.gate_print,
                                        output_dir=output_dir,
                                        logger=_logger,
                                        wandb_enable=args.wandb_enable)
        if rank == 0:
            # epoch = 1, csv write_head initialization
            update_summary(epoch, train_metrics, os.path.join(output_dir, 'summary_train.csv'), 
                           split='train', write_header=(epoch==1))
        
        if epoch % args.test_interval == 0:  # test per ${test_interval} epoch
            test_metrics, best_metrics = test_one_epoch(task_name=args.task,
                                                        model_name=args.model,
                                                        epoch=epoch,
                                                        total_epoch=args.epoch,
                                                        model=model,
                                                        device=device,
                                                        testloader=testloader,
                                                        output_dir=output_dir,
                                                        loss_dict=loss_dict,
                                                        best_metrics=best_metrics,
                                                        crop_ratio=crop_ratio,
                                                        overlap_crop=args.overlap_crop,
                                                        gate_print=args.gate_print,
                                                        logger=_logger,
                                                        wandb_enable=args.wandb_enable)
            if rank == 0:
                update_summary(epoch, test_metrics, os.path.join(output_dir, 'summary_test.csv'), 
                               split='test', write_header=(epoch==args.test_interval))
        
        if scheduler is not None:
            scheduler.step()

        dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout

    cleanup()


if __name__ == "__main__":
    main()