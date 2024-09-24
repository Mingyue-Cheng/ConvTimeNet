if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

export CUDA_VISIBLE_DEVICES="5"

dataset='PEMS_SF'
model_name='ConvTimeNet'

ps=16
sd=8
dw_ks='19,19,29,29,37,37'

python -u run_classification.py \
    --is_training 1 \
    --random_seed 2023 \
    --root_path datasets/$dataset \
    --model $model_name \
    --dataset $dataset \
    --patch_size $ps \
    --patch_stride $sd \
    --dw_ks $dw_ks \
    --d_model 64 \
    --d_ff 256 \
    --dropout 0.1 \
    --learning_rate 0.001 \
    --lr_decay 20 \
    --itr 3 \
    --train_epochs 100 >logs/$model_name'_'$dataset.log 