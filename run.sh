deepspeed --num_gpus=1        \
  train_dpo.py                \
  --deepspeed ds_config.json  \
  --use_wandb                 \
  --dim 512                   \
  --n_layers 8                \
  --use_moe=True