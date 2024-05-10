#################################### Allweather #######################################################
model_path=output/train/allweather_moe-film-linear-basenet-star-gelu-n4-k2_bs64_ep200_ps8_embed384_mlpx4_mlpupsample-outchx4_cnn-embed_wo-pe_normalize_vgg0.04_lr0.0002/best_metric.pth
output_dir=output/infer_one/allweather_moe-film-linear-basenet-star-gelu-n4-k2/derain
img_path=/data/lyl/code2/MoWE_DDP/output/infer/allweather_moe-film-linear-basenet-star-gelu-n4-k2/deraindrop/0_rain.png
dataset=allweather
task=derain
cuda='4,5,6,7'

mkdir -p ${output_dir}

export CUDA_VISIBLE_DEVICES=${cuda}

python infer_one.py --task ${task} --dataset ${dataset} \
--img-path $img_path \
--model-path $model_path --model-name mowe \
--gpu-list 4 5 6 7 \
--exp $output_dir \
>> ${output_dir}/exp.txt 2>&1