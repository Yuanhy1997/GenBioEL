# How to run our code:

## requirements

The python package required are as listed below, deepspeed is used for pretraining, if you only want to finetune on downstream tasks, please omit this one. For other omitted requirements, any compatible ones will do. 

|deepspeed | 0.5.4 |
|nltk  | 3.6.5 |
|numpy | 1.19.5 |
|scipy | 1.5.4 |
|torch | 1.9.1+cu111 |
|transformers | 4.11.3 |

## finetuning

1. Please put the preprocessed dataset under this directory, the code for data preprocessing is provided in ./data_utils/
2. Before run the bash files, please create the prefix tree for decoding first
   1. cd ./trie
   2. modify the KB path in create_trie_and_target_kb.py
   3. run the python file python3 create_trie_and_target_kb.py
   4. cd ..
3. run the bash file to finetune and test on different benchmarks

# pretraining

The pretraining code is provided in ./kb_guided_pretrain/ and is written with deepspeed using ZeRO2.

# Resources

The knowledge base pretrained checkpoint can be downloaded from this [link](https://drive.google.com/file/d/1TqvQRau1WPYE9hKfemKZr-9ptE-7USAH/view?usp=sharing),

The preprocessed datasets can be downloaded from this [link](https://drive.google.com/file/d/1JWYMdwxp7_ZZRGAO-ENmgUNirx9-nX32/view?usp=sharing).