export CUDA_VISIBLE_DEVICES=0;
export OMP_NUM_THREADS=1;
python -m torch.distributed.launch --nnodes=1 --master_port 12345 --nproc_per_node=2  train.py --name mm --stage test --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
    --pretrained_path "/home/data/Jingkai/alex/finetuning_outputs/nihxray/mm_bestauc_checkpoint.pth" --dataset_path '/home/data/Jingkai/alex/nihxray' \
    --eval_batch_size 512 --img_size 224 --fp16 --fp16_opt_level O2