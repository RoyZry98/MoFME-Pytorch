# ###################################  Baseline  ########################################################

# model_path=/data1/lyl/code/MoWE_DDP/output/test/maw-sim_baseline/dehaze/best_metric.pth
# output_dir=maw-sim_baseline
# dataset=maw_sim
# task=derain
# cuda=4

# export CUDA_VISIBLE_DEVICES=${cuda}

# python infer.py --task $task --dataset ${dataset} --split infer+test \
# --model-path $model_path --model-name mowe \
# --bs 4 \
# --gpu-list ${cuda} \
# --exp $output_dir

# ###################################  MoE Baseline  ########################################################

model_path=/data/lyl/code2/MoWE_DDP/output/train/allweather_moe-film-linear-basenet-star-gelu-n4-k2_bs64_ep200_ps8_embed384_mlpx4_mlpupsample-outchx4_cnn-embed_wo-pe_normalize_vgg0.04_lr0.0002/best_metric.pth
output_dir=allweather_moe-film-linear-basenet-star-gelu-n4-k2
mkdir -p output/infer/${output_dir}

dataset=allweather
task=deraindrop
cuda=7

export CUDA_VISIBLE_DEVICES=${cuda}

python infer.py --task $task --dataset ${dataset} --split infer+test \
--model-path $model_path --model-name mowe \
--bs 4 \
--gpu-list ${cuda} \
--exp $output_dir \
>> output/infer/${output_dir}/exp.txt 2>&1