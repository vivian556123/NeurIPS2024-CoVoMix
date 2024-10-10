TORCH_DISTRIBUTED_DEBUG=DETAIL 
python train.py --base_dir /home/overlap_pair_data/ \
 --no_wandb \
 --batch_size=8 \
 --gpus=8 \
 --format hubert_overlap_two_input_one_output \
 --train_subset train_separate \
 --cond_drop_prob 0.3 \
 --dev_base_dir /home/overlap_pair_data/ \
 --dev_subset val_separate \
 --fisher_data \
 --CoVoMix_model acoustic \
 --num_eval_files 5 \
 --CoVoMix_num_phoneme_tokens 502 \
 --lr_scheduler \
 --CoVoMix_dim 160 \
 --twocondition_oneoutput \
 --random_mask \
 --CoVoMix_depth 8 \
 --model_save_dir /exp/CoVoMix/VoMix \
