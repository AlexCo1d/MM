export CUDA_VISIBLE_DEVICES=0;
export OMP_NUM_THREADS=1;
python -m torch.distributed.launch --nnodes=1 --master_port 12345 --nproc_per_node=1  train.py --name mm --stage test -z --num_classes 5 \
    --pretrained_path "/home/data/Jingkai/alex/pretrain/checkpoint-99.pth" --dataset_path '/home/data/Jingkai/alex/chexpert' \
    --eval_batch_size 512 --img_size 224 --fp16 --fp16_opt_level O2