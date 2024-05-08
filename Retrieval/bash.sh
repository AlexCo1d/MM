export CUDA_VISIBLE_DEVICES=1,2;
python -m torch.distributed.launch --nnodes=1 --master_port 12345 --nproc_per_node=2 task.py \
--checkpoint \
--datapath