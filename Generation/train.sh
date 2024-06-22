export CUDA_VISIBLE_DEVICES=1,2;
export TORCH_DISTRIBUTED_DEBUG=DETAIL;
python -m torch.distributed.launch --nnodes=1 --master_port 22437 --nproc_per_node=2 train.py \
--num_workers 6 \
--batch_size 28  \
--epochs 40 \
--is_lora \
--eval_freq 5 \
--eval_batch_size 32 \
--warmup_epochs 8 \
--start_epoch 0 \
--lr 3e-5 \
--dataset_use iu --dataset_path /home/data/Jingkai/alex/iuxray \
--checkpoint /home/data/Jingkai/alex/weight/MMFormer-80.pth \
--LLM_path /home/data/Jingkai/alex/weight/pmc-llama  \
--output_dir /home/data/Jingkai/alex/gen_out_iu