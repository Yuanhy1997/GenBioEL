#!/bin/bash 

DEVICE_NUMBER=$1
MODEL_NAME=$2
INIT_MODEL=$3
DATASET=$4

for idx in {0..9};
do
CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/aap/fold$idx \
                                            -model_load_path $INIT_MODEL \
                                            -model_token_path facebook/bart-large \
                                            -model_save_path ./model_checkpoints/$MODEL_NAME/$idx \
                                            -save_steps 30000 \
                                            -logging_path ./logs/$MODEL_NAME/$idx \
                                            -logging_steps 100 \
                                            -init_lr 5e-6 \
                                            -per_device_train_batch_size 8 \
                                            -evaluation_strategy no \
                                            -label_smoothing_factor 0.1 \
                                            -max_grad_norm 0.1 \
                                            -max_steps 30000 \
					                        -warmup_steps 500 \
                                            -weight_decay 0.01 \
                                            -lr_scheduler_type polynomial \
                                            -attention_dropout 0.1  \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -max_position_embeddings 1024 \
					                        -rdrop 0 \
                                            -seed 0 \
					                        -finetune \
					                        -prefix_mention_is 
done

for idx in {0..9};
do
CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/aap/fold$idx \
                                            -evaluation \
					                        -seed 0 \
                                            -model_load_path ./model_checkpoints/$MODEL_NAME/$idx/checkpoint-30000 \
                                            -model_token_path facebook/bart-large \
                                            -trie_path $DATASET/aap/trie.pkl \
					                        -dict_path $DATASET/aap/target_kb.json \
                                            -per_device_eval_batch_size 1 \
                                            -max_position_embeddings 1024 \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -num_beams 5 \
                                            -max_length 1024 \
                                            -min_length 1 \
                                            -dropout 0.1 \
                                            -attention_dropout 0.1 \
                                            -prefix_prompt \
                                            -prefix_mention_is
done
