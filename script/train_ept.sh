#!/bin/bash
# EPT Training Script
# Expert kernel sizes: 2 2 4 4 6 6 8 8

OUTPUT_MODEL="./save/EPT_22446688_topk2"
mkdir -p "$OUTPUT_MODEL"

python -c "from transformers import T5Tokenizer; T5Tokenizer.from_pretrained('/root/data/t5-base'); print('[tok] initialized')"

nohup env NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 HF_HUB_OFFLINE=1 deepspeed --include localhost:0 --master_port 17621 /root/EPT/finetune.py \
  --model_name_or_path /root/data/t5-base \
  --tasks cola mnli mrpc qnli qqp rte sst2 stsb \
  --max_length 128 \
  --use_lora True \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules q k v o wi wo \
  --expert_kernel_sizes 2 2 4 4 6 6 8 8 \
  --moe_top_k 2 \
  --output_dir "$OUTPUT_MODEL" \
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 512 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --logging_dir "$OUTPUT_MODEL/logs" \
  --logging_steps 100 \
  --load_best_model_at_end True \
  --dataloader_num_workers 16 \
  --bf16 True \
  --seed 2023 > "$OUTPUT_MODEL/train.log" 2>&1 &

echo "EPT training started in background"