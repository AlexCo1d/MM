export CUDA_VISIBLE_DEVICES=1,2,3;
python -m torch.distributed.launch --nnodes=1 --master_port 12345 --nproc_per_node=3 VQA_RAD/train.py \
--dataset_use vqarad --dataset_path /home/data/Jingkai/alex/radvqa/ \
--checkpoint /home/data/Jingkai/alex/pretrain_base1/checkpoint-210.pth \
--output_dir /home/data/Jingkai/alex/vqa_out