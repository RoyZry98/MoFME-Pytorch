import os
import sys
import argparse
import logging
import time

import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import json
from collections import OrderedDict
import wandb

# utils
from utils import AverageMeter, mprint

def train_one_epoch(task_name, epoch, total_epoch, model_name, model, moe_enable, gate, trainloader, 
                    optimizer, device,  loss_dict, gate_print, logger, output_dir,
                    wandb_enable, log_every=40, *args, **kwargs):
    mprint('\n')
    metrics = {
        'train_loss': AverageMeter(),
        'aux_loss': AverageMeter(),
    }
    examples = []
    rank = dist.get_rank()
    running_loss = 0
    log_steps = 0
    start_time = time.time()

    model.train()

    trainloader.sampler.set_epoch(epoch)
    mprint(f"Beginning epoch {epoch} | {total_epoch}...")

    for batch_idx, (inputs, targets, task_idx) in enumerate(trainloader):

        optimizer.zero_grad()

        ''' intputs '''
        inputs = inputs.to(device)

        ''' targets '''
        targets = targets.to(device)

        ''' task idx'''
        task_idx = task_idx.to(device)

        ''' forward '''
        outputs, cls_outputs, weights_list, l_aux = model(inputs, task_idx)

        if gate_print and batch_idx % 40 == 0 and rank == 0:
            task_idx_0 = int(task_idx[0].cpu().item())
            task_idx_1 = int(task_idx[1].cpu().item())
            mprint(task_idx_0, weights_list[0][0])
            mprint(task_idx_1, weights_list[0][1])
            mprint(task_idx_0, weights_list[1][0])
            mprint(task_idx_1, weights_list[1][1])

        ''' loss '''
        loss = 0
        total_aux_loss = 0
        # auxiliary loss, n experts
        if l_aux is not None:
            loss += l_aux.mean()

        # task loss
        for _function in loss_dict[task_name]:
            for _name in loss_dict[task_name][_function]:
                _loss = loss_dict[task_name][_function][_name]['loss'].to(device)
                _weight = loss_dict[task_name][_function][_name]['weight']
                
                loss += _weight * _loss(outputs, targets)

        # clip grad
        torch.nn.utils.clip_grad_norm_(parameters=model.module.parameters(), max_norm=10, norm_type=2)
        loss.backward(retain_graph=True)
        optimizer.step()

        if wandb_enable and rank == 0:
            wandb.log({'epoch': epoch, 'loss': loss})

        metrics['train_loss'].update(loss.item())
        metrics['aux_loss'].update(total_aux_loss.item() if torch.is_tensor(total_aux_loss) else total_aux_loss)

        # Log loss values:
        running_loss += loss.item()
        log_steps += 1
        if batch_idx % log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            mprint(f"(step={batch_idx:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time.time()

        if wandb_enable and batch_idx == 0 and rank == 0:
            inputs_wandb = inputs[0]
            outputs_wandb = outputs[0]
            targets_wandb = targets[0]
            examples.append(wandb.Image(inputs_wandb, caption="Img Input"))
            examples.append(wandb.Image(outputs_wandb, caption="Img Output"))
            examples.append(wandb.Image(targets_wandb, caption="{} Ground Truth".format(task_name)))

    train_loss = metrics['train_loss'].avg
    aux_loss = metrics['aux_loss'].avg
    lr_current = optimizer.state_dict()['param_groups'][0]['lr']
    recon_loss = round(train_loss - aux_loss, 6)
    if rank == 0:
        logger.info('Epoch:{} total:{} recon: {} aux: {} lr:{}\n'.format(
                    epoch, train_loss, recon_loss, aux_loss, round(lr_current, 7)))

    # wandb setting
    if wandb_enable and rank == 0:
        wandb.log({
            "Examples_Train": examples,
        })

    # csv
    train_metrics = OrderedDict([('loss', train_loss), ('recon_loss', recon_loss)])

    # save pth
    state = {
        'net': model.module if (torch.cuda.is_available()) else model,
        'arch': model_name,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, output_dir + '/last_metric.pth')

    return train_metrics
