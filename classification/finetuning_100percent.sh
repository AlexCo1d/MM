CUDA_VISIBLE_DEVICES=1 python3 train.py --name mm --stage train --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
    --pretrained_path "/home/data/Jingkai/alex/pretrain/checkpoint-99.pth" --dataset_path '/home/data/Jingkai/alex/nihxray' \
    --output_dir "/home/data/Jingkai/alex/finetuning_outputs/nihxray" --data_volume '100' --num_steps 200000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 5000 --fp16 --fp16_opt_level O2 --train_batch_size 96

