import os
import numpy as np
from tqdm import tqdm
import json
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from einops import repeat
import wandb

# metric
from piq import psnr as PSNR
from piq import ssim as SSIM

# utils
from utils import AverageMeter, mprint

def test_one_epoch(task_name, model_name, epoch, total_epoch, model, device, testloader,
                   output_dir, loss_dict, best_metrics, crop_ratio, overlap_crop, 
                   gate_print, logger, wandb_enable, *args, **kwargs):

    model.eval()

    metrics = {
        'test_loss': AverageMeter(),
        'psnr': AverageMeter(),
        'ssim': AverageMeter(),
    }

    rank = dist.get_rank()
    running_loss = 0
    log_steps = 0

    if rank == 0:
        logger.info('Best_Epoch:{} test_psnr: {} test_ssim: {}'.format(
                    best_metrics[2], best_metrics[0], best_metrics[1]))
    best_psnr, best_ssim, best_epoch = best_metrics

    examples = []

    weights_l1 = 0
    weights_l2 = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, task_idx) in enumerate(testloader):

            ''' inputs '''
            inputs = inputs.to(device)
            count_inputs = torch.zeros_like(inputs)

            ''' targets '''
            targets = targets.to(device)

            ''' task idx'''
            task_idx = task_idx.to(device)

            outputs = torch.zeros_like(targets)
            count_outputs = torch.zeros_like(targets)

            # ''' crop '''
            h, w = inputs.shape[-2], inputs.shape[-1]
            h_out, w_out = targets.shape[-2], targets.shape[-1]
            # d: delta
            h_d, w_d = int(h * crop_ratio[0]), int(w * crop_ratio[1])
            h_d_out, w_d_out = int(h_out * crop_ratio[0]), int(w_out * crop_ratio[1])
            num_h = int(1.0/crop_ratio[0])
            num_w = int(1.0/crop_ratio[1])
            if not overlap_crop:  # no overlap
                for i in range(num_h):
                    for j in range(num_w):
                        if i == 0 and j == 0:
                            inputs_crop = inputs[:, :, i*h_d: (i+1)*h_d, j*w_d: (j+1)*w_d]
                        else:
                            inputs_crop = torch.cat([inputs_crop, inputs[:, :, i*h_d: (i+1)*h_d, j*w_d: (j+1)*w_d]], dim=0)
                task_idx = repeat(task_idx, 'b -> (r b)', r=num_h*num_w)
            else:  # overlap
                for i in np.arange(0, num_h-0.5, 0.5):
                    for j in np.arange(0, num_w-0.5, 0.5):
                        i, j = float(i), float(j)
                        if i == 0.0 and j == 0.0:
                            inputs_crop = inputs[:, :, 0:h_d, 0:w_d]
                            count_inputs[:, :, 0:h_d, 0:w_d] += 1
                            continue
                        else:
                            inputs_crop_temp = inputs[:, :, int(i*h_d): int((i+1)*h_d), int(j*w_d): int((j+1)*w_d)]
                            count_inputs[:, :, int(i*h_d):int((i+1)*h_d), int(j*w_d):int((j+1)*w_d)] += 1
                            inputs_crop = torch.cat([inputs_crop, inputs_crop_temp], dim=0)
                task_idx = repeat(task_idx, 'b -> (r b)', r=(2*num_h-1)*(2*num_w-1))
            _outputs, cls_outputs, weights_list, l_aux = model(inputs_crop, task_idx)

            if weights_list != [] and weights_list is not None:
                weights_l1 += torch.mean(weights_list[0], dim=0)
                weights_l2 += torch.mean(weights_list[1], dim=0)

            bs = outputs.shape[0]
            if not overlap_crop:  # no overlap
                for i in range(num_h):
                    for j in range(num_w):
                        outputs[:, :, i*h_d_out: (i+1)*h_d_out, j*w_d_out: (j+1)*w_d_out] = \
                            _outputs[(i*num_w+j)*bs:(i*num_w+j+1)*bs, :, :, :]
            else:  # overlap
                for _i, i in enumerate(np.arange(0, num_h-0.5, 0.5)):  # _i: 个数, i: 位置
                    for _j, j in enumerate(np.arange(0, num_w-0.5, 0.5)):
                        i, j = float(i), float(j)
                        outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += \
                            _outputs[int((_i*(2*num_w-1)+_j)*bs): int((_i*(2*num_w-1)+_j+1)*bs), :, :, :]
                        count_outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += 1
                outputs = torch.div(outputs, count_outputs)

            loss = 0
            if l_aux is not None:
                loss += torch.mean(l_aux)
            for _function in loss_dict[task_name]:
                for _name in loss_dict[task_name][_function]:
                    _loss = loss_dict[task_name][_function][_name]['loss']
                    _weight = loss_dict[task_name][_function][_name]['weight']

                    img_loss = _weight * _loss(outputs, targets)
                    loss += img_loss

            # evaluation
            metrics['test_loss'].update(round(loss.item(), 5))
            outputs = torch.clamp(outputs, 0, 1)
            # outputs = unnormalize(outputs)
            # targets = unnormalize(targets)

            psnr = PSNR(x=outputs, y=targets).item()
            ssim = SSIM(x=outputs, y=targets).item()

            metrics['psnr'].update(psnr)
            metrics['ssim'].update(ssim)

            _psnr = metrics['psnr'].avg
            _ssim = metrics['ssim'].avg

            # Log images in your test dataset automatically,
            # along with predicted and true labels by passing pytorch tensors with image data into wandb.
            if wandb_enable and batch_idx == 0 and rank == 0:
                examples.append(wandb.Image(
                    inputs[0],
                    caption="Input"))
                examples.append(wandb.Image(
                    outputs[0],
                    caption="Output"))
                examples.append(wandb.Image(
                    targets[0],
                    caption="Ground Truth"))

    if gate_print and weights_list != [] and weights_list is not None and rank == 0:
        mprint('w1:', weights_l1)
        mprint('w2:', weights_l2)

    test_loss = metrics['test_loss'].avg
    psnr, ssim = metrics['psnr'].avg, metrics['ssim'].avg
    if rank == 0:
        logger.info('Epoch:{} test_loss:{}'.format(epoch, test_loss))
        logger.info('Epoch:{} test_psnr:{} test_ssim:{}'.format(epoch, psnr, ssim))

    # Save checkpoint.
    _psnr, _ssim = metrics['psnr'].avg, metrics['ssim'].avg
    if _psnr >= best_psnr and _ssim >= best_ssim and rank == 0:

        logger.info('Saving..')

        best_ssim = _ssim
        best_psnr = _psnr
        best_epoch = epoch

        state = {
            'net': model.module if (torch.cuda.is_available()) else model,
            'arch': model_name,
            'model_state_dict': model.state_dict(),
            'epoch': best_epoch,
            'psnr': best_psnr,
            'ssim': best_ssim,
        }

        torch.save(state, output_dir + '/best_metric.pth')

        # json
        metric_dict = {
            'best_epoch': best_epoch,
        }

        metric_img_dict = {
            'best_test_psnr': psnr,
            'best_test_ssim': ssim
        }
        metric_dict.update(metric_img_dict)

        json_file = os.path.join(output_dir, "best.json")
        with open(json_file, 'w') as f:
            json.dump(metric_dict, f, indent=2)

        logger.info('best_psnr:{} best_ssim:{} best_epoch:{}\n'.format(best_psnr, best_ssim, best_epoch))

    # wandb setting
    if wandb_enable and rank == 0:
        wandb.log({
            "Examples_Test": examples,
            "Test Loss": test_loss,
            "Test PSNR": _psnr,
            "Test SSIM": _ssim,
        })

    # csv
    test_metrics_list = [('loss', test_loss)]
    test_metrics_list += [('psnr', psnr), ('ssim', ssim)]
    test_metrics = OrderedDict(test_metrics_list)

    # best_metrics
    best_metrics = (best_psnr, best_ssim, best_epoch)

    return test_metrics, best_metrics


