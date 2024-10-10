TORCH_DISTRIBUTED_DEBUG=DETAIL 
python train.py --base_dir /home/train/ \
 --no_wandb \
 --batch_size=6 \
 --gpus=8 \
 --format hubert_fisher \
 --train_subset train \
 --cond_drop_prob 0.3 \
 --dev_base_dir /home/train/ \
 --dev_subset val \
 --fisher_data \
 --CoVoMix_model acoustic \
 --num_eval_files 5 \
 --CoVoMix_depth 8 \
 --CoVoMix_num_phoneme_tokens 502 \
 --model_save_dir /exp/CoVoMix/VoSingle \
