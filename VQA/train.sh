export CUDA_VISIBLE_DEVICES=1,2,3;
python -m torch.distributed.launch --nnodes=1 --master_port 13355 --nproc_per_node=3 train.py \
--num_workers 6 \
--batch_size 32 \
--epochs 40 \
--eval_freq 5 \
--is_lora \
--start_epoch 0 \
--warmup_epochs 8 \
--img_size 384 \
--eval_batch_size 32 \
--min_lr 2e-6 \
--dataset_use radvqa --dataset_path /home/data/Jingkai/alex/radvqa \
--checkpoint /home/data/Jingkai/alex/vqa_out_rad/ \
--LLM_path /home/data/Jingkai/alex/weight/pmc-llama \
--output_dir /home/data/Jingkai/alex/vqa_out_rad