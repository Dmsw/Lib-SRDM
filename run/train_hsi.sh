CUDA_VISIBLE_DEVICES=6, python ../scripts/hsi_train.py \
    --data_dir "/home/root/dataset/NTIRE22" \
    --batch_size 8 \
    --save_interval 20000 \
    --model_config ../config/model_config.yaml \
    --lr_anneal_steps 100000 \
    --save_dir ../models/icvl/