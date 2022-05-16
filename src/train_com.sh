

DEVICE_NUMBER=$1
DATASET=$2
MODEL_NAME=$3
INIT_MODEL=$4

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/cometa/ \
                                            -model_load_path $INIT_MODEL \
                                            -model_save_path ./model_checkpoints/$MODEL_NAME \
                                            -model_token_path facebook/bart-large \
                                            -save_steps 40000 \
                                            -logging_path ./logs/$MODEL_NAME \
                                            -logging_steps 50 \
                                            -init_lr 1e-05 \
                                            -per_device_train_batch_size 8 \
					                        -rdrop 0.0 \
                                            -evaluation_strategy no \
                                            -label_smoothing_factor 0.1 \
                                            -max_grad_norm 0.1 \
                                            -max_steps 40000 \
					                        -warmup_steps 500 \
                                            -weight_decay 0.01 \
                                            -lr_scheduler_type polynomial \
                                            -attention_dropout 0.1  \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -max_position_embeddings 1024 \
                                            -seed 0 \
					                        -finetune \
					                        -prefix_mention_is

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/cometa \
                                            -evaluation \
                                            -trie_path $DATASET/cometa/trie.pkl\
                                            -per_device_eval_batch_size 1 \
                                            -max_position_embeddings 1024 \
					                        -model_load_path ./model_checkpoints/$MODEL_NAME/checkpoint-40000 \
                                            -model_token_path facebook/bart-large \
					                        -seed 0 \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -prefix_prompt \
                                            -num_beams 5 \
                                            -max_length 1024 \
                                            -min_length 1 \
                                            -dropout 0.1 \
                                            -attention_dropout 0.1 \
					                        -dict_path $DATASET/cometa/target_kb.json \
                                            -prefix_mention_is \
