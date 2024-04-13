export CUDA_VISIBLE_DEVICES=0,2,3;
export OMP_NUM_THREADS=1;
python -m torch.distributed.launch --nnodes=1 --master_port 12345 --nproc_per_node=3 --use_env main_pretrain.py \
    --num_workers 10 \
    --accum_iter 2 \
    --batch_size 64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 2e-4 --weight_decay 0.05 \
    --min_lr 1e-7 \
    --resume /home/data/Jingkai/alex/weight/MM.pth \
    --data_path /home/data/Jingkai/alex/mimic \
    --output_dir /home/data/Jingkai/alex/pretrain
~