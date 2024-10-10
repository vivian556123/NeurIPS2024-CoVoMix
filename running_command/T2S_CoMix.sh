CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  TORCH_DISTRIBUTED_DEBUG=DETAIL export CUDA_LAUNCH_BLOCKING=1 
python train.py --base_dir /home/overlap_fisher_text2semantic \
 --no_wandb \
 --batch_size=6 \
 --gpus=8 \
 --format text2semantic_2output \
 --train_subset train \
 --dev_base_dir /home/overlap_fisher_text2semantic \
 --dev_subset val \
 --fisher_data \
 --CoVoMix_model text2semantic \
 --num_eval_files 5 \
 --text2semantic \
 --text2semantic_data \
 --CoVoMix_dim_transformer 512 \
 --laughter_tokenizer \
 --text2semantic_tokens 501 \
 --text2semantic_two_output \
 --target_transformer_dim 1024 \
 --model_save_dir /exp/CoVoMix/CoMix  \