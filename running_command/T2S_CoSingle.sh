TORCH_DISTRIBUTED_DEBUG=DETAIL export CUDA_LAUNCH_BLOCKING=1 
python train.py --base_dir /home/Fisher_English_Processed_Conversation/ \
 --no_wandb \
 --batch_size=10 \
 --gpus=8 \
 --format text2semantic \
 --train_subset train \
 --dev_base_dir  /home/Fisher_English_Processed_Conversation \
 --dev_subset val \
 --fisher_data \
 --CoVoMix_model text2semantic \
 --num_eval_files 5 \
 --text2semantic \
 --text2semantic_data \
 --CoVoMix_dim_transformer 512 \
 --laughter_tokenizer \
 --text2semantic_tokens 501 \
 --model_save_dir /exp/CoVoMix/CoSingle  \
