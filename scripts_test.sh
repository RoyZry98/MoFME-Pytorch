################################### Allweather ##############################
test_dir=/data/lyl/code2/MoWE_DDP/output/test
mkdir -p ${test_dir}
num_experts=(4)
top_k=(2)
tasks=(derain desnow deraindrop)
for n in "${num_experts[@]}"
do
    for k in "${top_k[@]}"
    do
        for task in "${tasks[@]}"
        do
            dataset=allweather
            output_dir=allweather_moe-film-linear-basenet-star-gelu-n${n}-k${k}_ep200
            model_path=output/train/allweather_moe-film-linear-basenet-star-gelu-n${n}-k${k}_bs64_ep200_ps8_embed384_mlpx4_mlpupsample-outchx4_cnn-embed_wo-pe_normalize_vgg0.04_lr0.0002/best_metric.pth
            cuda=5
            export CUDA_VISIBLE_DEVICES=${cuda}

            python test.py \
            --task $task --dataset $dataset --split test \
            --model-path $model_path --model-name mowe \
            --overlap-crop \
            --loss-list content l1 1 \
            --bs 4 \
            --gpu-list ${cuda} \
            --exp $output_dir
        done
        python statistics.py --output ${test_dir}/${output_dir}
    done
done