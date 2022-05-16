deepspeed --include localhost:0,1,2,3,4,5 \
          ./train.py --config-file ./bart.json \
                     --output_dir ./synonyms_pretrained_model \
                     --token_nosing_prob 0.1 \
		             --label_smoothing_factor 0.1 \
                     --max_seq_length 1024 \
                     --max_predictions_per_seq 150 \
                     --seed 42 \
                     --lr_schedule LL \
                     --job_name st21pv_pretrain \
                     --print_steps 10 \
                     --save_steps 100 \
                     --data_path_prefix ./datagen \
                     --deepspeed --deepspeed_config ./ds_config_zero2.json
