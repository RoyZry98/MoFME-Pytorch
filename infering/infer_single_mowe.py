import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision.utils import save_image

def infer_one_epoch(model, device, testloader, output_dir, crop_ratio, logger, 
                    dataset_name, *args, **kwargs):
    model.eval()

    loader_tqdm = tqdm(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets, input_name) in enumerate(loader_tqdm):

            ''' inputs '''
            inputs = inputs.to(device)
            count_inputs = torch.zeros_like(inputs)

            ''' targets '''
            targets = targets.to(device)

            outputs = torch.zeros_like(targets)
            count_outputs = torch.zeros_like(targets)

            ''' crop '''
            h, w = inputs.shape[-2], inputs.shape[-1]
            h_out, w_out = targets.shape[-2], targets.shape[-1]
            # d: delta
            h_d, w_d = int(h * crop_ratio[0]), int(w * crop_ratio[1])
            h_d_out, w_d_out = int(h_out * crop_ratio[0]), int(w_out * crop_ratio[1])
            num_h = int(1.0/crop_ratio[0])
            num_w = int(1.0/crop_ratio[1])
            # for i in range(num_h):  # no overlap
            #     for j in range(num_w):
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

            task_idx = None
            _outputs, cls_outputs, weights_list, l_aux = model(inputs_crop, task_idx)

            bs = outputs.shape[0]
            # for i in range(num_h):  # no overlap
            #     for j in range(num_w):
            for _i, i in enumerate(np.arange(0, num_h-0.5, 0.5)):  # _i: 个数, i: 位置
                for _j, j in enumerate(np.arange(0, num_w-0.5, 0.5)):
                    i, j = float(i), float(j)
                    outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += \
                        _outputs[int((_i*(2*num_w-1)+_j)*bs): int((_i*(2*num_w-1)+_j+1)*bs), :, :, :]
                    count_outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += 1
            outputs = torch.div(outputs, count_outputs)

            loader_tqdm.set_description(f'Inference Epoch')

            # save
            for i in range(bs):
                _input_name = input_name[i]
                if dataset_name == 'cityscapes':
                    scene_name = _input_name.split("_")[0]
                    os.makedirs(output_dir,exist_ok=True)
                    os.makedirs(os.path.join(output_dir,scene_name),exist_ok=True)
                    save_image(outputs[0], os.path.join(output_dir, scene_name, _input_name))
                else:
                    save_image(outputs[i], os.path.join(output_dir, _input_name))

        #     # analysis
        #     if batch_idx == 0:
        #         weight_0 = weights_list[0].mean(dim=0).mean(dim=0).unsqueeze(dim=0)
        #         weight_1 = weights_list[1].mean(dim=0).mean(dim=0).unsqueeze(dim=0)
        #     else:
        #         weight_0 = torch.cat([weight_0, weights_list[0].mean(dim=0).mean(dim=0).unsqueeze(dim=0)], dim=0)
        #         weight_1 = torch.cat([weight_1, weights_list[1].mean(dim=0).mean(dim=0).unsqueeze(dim=0)], dim=0)

        # print(torch.round(weight_0.mean(dim=0).detach().cpu(), decimals=4))
        # print(torch.round(weight_1.mean(dim=0).detach().cpu(), decimals=4))

        #     # save
        #     if batch_idx == 0:
        #         weight_0 = weights_list[0].mean(dim=1)  # [b, n, d] -> [b, d], 把子图1024取平均, 子图level统计
        #         weight_1 = weights_list[1].mean(dim=1)
        #     else:
        #         weight_0 = torch.cat([weight_0, weights_list[0].mean(dim=1)], dim=0)
        #         weight_1 = torch.cat([weight_1, weights_list[1].mean(dim=1)], dim=0)

        # print(weight_0.shape)
        # np.save(os.path.join(output_dir, 'layer_0.npy'), weight_0.cpu().detach().numpy())
        
        # print(weight_1.shape)
        # np.save(os.path.join(output_dir, 'layer_1.npy'), weight_1.cpu().detach().numpy())

        logger.info("==> Finish inference, the outputs have been saved in {}".format(output_dir))

