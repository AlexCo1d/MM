export CUDA_VISIBLE_DEVICES=0,1;
python -m torch.distributed.launch --master_port 12345 --nproc_per_node=2 main_pretrain.py \
    --num_workers 10 \
    --accum_iter 2 \
    --batch_size 32 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --resume /home/data/Jingkai/alex/weight/MM.pth \
    --data_path /home/data/Jingkai/alex/mimic \
    --output_dir /home/data/Jingkai/alex/pretrain