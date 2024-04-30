export CUDA_VISIBLE_DEVICES=1,2,3;
python -m torch.distributed.launch --nnodes=1 --master_port 12345 --nproc_per_node=3 train.py \
--batch_size 28 \
--dataset_use radvqa --dataset_path /home/data/Jingkai/alex/radvqa/ \
--checkpoint /home/data/Jingkai/alex/pretrain_base1/checkpoint-200.pth \
--LLM_path /home/data/Jingkai/alex/weight/  \
--output_dir /home/data/Jingkai/alex/vqa_out