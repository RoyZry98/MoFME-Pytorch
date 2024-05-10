import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor, Normalize
from torchvision.utils import save_image

import os

import argparse
import numpy as np
import random
from PIL import Image

from configs import get_img_size, get_crop_ratio
from utils import Recorder

def main():
    """
    main()
    """

    parser = argparse.ArgumentParser(description='MoWE Inference for One Image')

    ######### Basic Setting ##############

    parser.add_argument('--task', default='derain', type=str, help='task name')
    parser.add_argument('--dataset', default='mawsim', type=str, help='dataset name')
    parser.add_argument('--img-path', type=str, required=True, help='image path')
    parser.add_argument('--img-size', type=int, nargs='+', default=None, help='img size')
    parser.add_argument('--crop-ratio', type=float, nargs='+', default=None, help='crop ratio')
    parser.add_argument('--model-path', default=None, type=str, help='inference model path')
    parser.add_argument('--model-name', default='mowe', type=str, help='inference model name')
    parser.add_argument('--exp', default='infer', type=str, help='output dir')

    ######### Visualization Setting ################

    parser.add_argument('--visualization', default=None, choices=['attn'], 
                                            type=str, help='Visualization choice')

    ######### GPU Setting ################

    parser.add_argument('--gpu-list', type=int, nargs='+', default=None, help='cuda list')

    ######### Inference Setting ################

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

    # output_dir = os.getcwd() + '/output/infer/' + args.exp
    output_dir = args.exp  # complete output dir path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('==> Output dir:')
    print(output_dir)

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Model
    print('\n==> Building model..')
    print('\n==> Load inference model from {}..'.format(args.model_path))
    model = torch.load(args.model_path, map_location=torch.device('cpu'))['net']
    if args.visualization == 'attn':
        model = Recorder(model)

    model = model.to(device)
    if 'cuda' in device:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        cudnn.benchmark = True
    model.eval()
    
    # Data
    print('\n==> Preparing data..')
    if args.img_size is not None:
        img = Image.open(args.img_path).convert('RGB').resize(args.img_size)
    else:  # dataset default
        img = Image.open(args.img_path).convert('RGB')
    input_name = args.img_path.split('/')[-1].replace('_img', '_img_{}'.format(args.task))
    
    to_tensor = ToTensor()
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    inputs = normalize(to_tensor(img)).unsqueeze(dim=0)  # [1, 3, h, w]

    # Infer
    ''' inputs '''
    inputs = inputs.to(device)
    count_inputs = torch.zeros_like(inputs)

    ''' outputs '''
    outputs = torch.zeros_like(inputs)
    count_outputs = torch.zeros_like(inputs)

    ''' crop '''
    h, w = inputs.shape[-2], inputs.shape[-1]
    h_out, w_out = outputs.shape[-2], outputs.shape[-1]

    # d: delta
    if args.crop_ratio is not None:
        crop_ratio = [1.0/args.crop_ratio[1], 1.0/args.crop_ratio[0]]
    else:  # dataset default
        crop_ratio = get_crop_ratio(dataset_name=args.dataset)
    h_d, w_d = int(h * crop_ratio[0]), int(w * crop_ratio[1])
    h_d_out, w_d_out = int(h_out * crop_ratio[0]), int(w_out * crop_ratio[1])
    num_h = int(1.0/crop_ratio[0])
    num_w = int(1.0/crop_ratio[1])
    for i in np.arange(0, num_h-0.5, 0.5):  # 2n-1
        for j in np.arange(0, num_w-0.5, 0.5):  # 2n-1
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
    if args.visualization == 'attn':
        (_outputs, cls_outputs, weights_list, l_aux), attns = model(inputs_crop, task_idx)
        print(attns.shape)
    else:
        _outputs, cls_outputs, weights_list, l_aux = model(inputs_crop, task_idx)

    bs = outputs.shape[0]
    for _i, i in enumerate(np.arange(0, num_h-0.5, 0.5)):  # _i: 个数, i: 位置
        for _j, j in enumerate(np.arange(0, num_w-0.5, 0.5)):
            i, j = float(i), float(j)
            outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += \
                _outputs[int((_i*(2*num_w-1)+_j)*bs): int((_i*(2*num_w-1)+_j+1)*bs), :, :, :]
            count_outputs[:, :, int(i*h_d_out): int((i+1)*h_d_out), int(j*w_d_out): int((j+1)*w_d_out)] += 1
    outputs = torch.div(outputs, count_outputs)

    # save
    save_image(outputs[0], os.path.join(output_dir, input_name))

    # analyse
    # print(weights_list[0].shape)  # [35, 1024, 16] = [(2h-1)*(2w-1), h*w/p*p, num_expert]
    # print(weights_list[1].shape)

    # torch.set_printoptions(precision=4, sci_mode=False)
    # weight_0 = weights_list[0].mean(dim=0).mean(dim=0)
    # print(torch.round(weight_0.detach().cpu(), decimals=4))

    # weight_1 = weights_list[1].mean(dim=0).mean(dim=0)
    # print(torch.round(weight_1.detach().cpu(), decimals=4))

if __name__ == '__main__':
    main()