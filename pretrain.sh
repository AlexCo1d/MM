CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port 11451 main_pretrain.py \
    --num_workers 10 \
    --accum_iter 2 \
    --batch_size 32 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --resume /home/data/Jingkai/alex/weight/MM.pth \
    --data_path /home/Jingkai/alex/MM \
    --output_dir /home/data/Jingkai/alex/pretrain \