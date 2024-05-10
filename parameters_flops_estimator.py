import torch
from ptflops import get_model_complexity_info
from models.mowe import MoWE
import time

if __name__ == '__main__':
    h = 240
    w = 240
    model = MoWE(task_scale='1+1+1+1',
            is_single_task=False,

            img_size=(h, w),
            patch_size=8,

            residual_learning=False,
            embed_dim=384,
            depth=2,
            scale_dim=4,

            moe_enable=True,
            num_expert=128,
            type_expert='ffn',
            top_k=4,
            gate='film',)
    model.eval()

    flops, params = get_model_complexity_info(model, (3, h, w), as_strings=True, print_per_layer_stat=True)

    print('Flops:  ' + flops)
    print('Params: ' + params)
    
    t = 0
    img = torch.randn(size=(1, 3, h, w)).cuda()
    model = model.cuda()
    for i in range(220):
        if i >= 20:
            t1 = time.time()
        output = model(img)
        # print(output)
        if i >= 20:
            t2 = time.time()
            t += (t2-t1)
    print('Times per image: {}'.format(t/200))
