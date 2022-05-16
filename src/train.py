import os
import sys

import numpy as np
from tqdm import tqdm
import pickle
import argparse

import torch 
import torch.nn as nn
from transformers import TrainingArguments
from trainer import modifiedSeq2SeqTrainer
from trie import Trie
from utils import reform_input
import copy
import json
import ipdb

def train(config):
    config.max_steps = config.max_steps // config.gradient_accumulate
    config.save_steps = config.max_steps
    training_args = TrainingArguments(
                    output_dir=config.model_save_path,          # output directory
                    num_train_epochs=config.num_train_epochs,              # total number of training epochs
                    per_device_train_batch_size=config.per_device_train_batch_size,  # batch size per device during training
                    per_device_eval_batch_size=config.per_device_eval_batch_size,   # batch size for evaluation
                    warmup_steps=config.warmup_steps,                # number of warmup steps for learning rate scheduler
                    weight_decay=config.weight_decay,               # strength of weight decay
                    logging_dir=config.logging_path,            # directory for storing logs
                    logging_steps=config.logging_steps,
                    save_steps=config.save_steps,
                    evaluation_strategy=config.evaluation_strategy,
                    learning_rate=config.init_lr,
                    label_smoothing_factor=config.label_smoothing_factor,
                    max_grad_norm=config.max_grad_norm,
                    max_steps=config.max_steps,
                    lr_scheduler_type=config.lr_scheduler_type,
                    seed=config.seed,
                    gradient_accumulation_steps=config.gradient_accumulate, 
                    )
    if config.t5:

        from models import T5EntityPromptModel
        from transformers import T5Tokenizer, T5Config
        from datagen import prepare_trainer_dataset_t5 as prepare_trainer_dataset

        t5conf = T5Config.from_pretrained('./t5-large')
        t5conf.dropout_rate = config.dropout

        tokenizer = T5Tokenizer.from_pretrained('./t5-large')

        model = T5EntityPromptModel.from_pretrained(config.model_load_path, 
                                                    config = t5conf,
                                                    finetune = config.finetune, 
                                                    n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                    load_prompt = config.load_prompt,
                                                    soft_prompt_path = config.model_load_path,
                                                    initialize_from_vocab = config.init_from_vocab,
                                                    )
        
    else:

        from models import BartEntityPromptModel
        from transformers import BartTokenizer, BartConfig
        # from datagen import prepare_trainer_dataset_genre as prepare_trainer_dataset
        if config.syn_pretrain:
            from datagen import prepare_trainer_dataset_pre as prepare_trainer_dataset
        else:
            if config.sample_train:
                from datagen import prepare_trainer_dataset_fine_sample as prepare_trainer_dataset
            else:
                from datagen import prepare_trainer_dataset_fine as prepare_trainer_dataset

        bartconf = BartConfig.from_pretrained(config.model_load_path)
        bartconf.max_position_embeddings = config.max_position_embeddings
        bartconf.attention_dropout = config.attention_dropout
        bartconf.dropout = config.dropout

        tokenizer = BartTokenizer.from_pretrained(config.model_token_path, 
                                                max_length=1024,
                                                )

        model = BartEntityPromptModel.from_pretrained(config.model_load_path, 
                                                    config = bartconf,
                                                    finetune = config.finetune, 
                                                    n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                    load_prompt = config.load_prompt,
                                                    soft_prompt_path = config.model_load_path,
                                                    no_finetune_decoder = config.no_finetune_decoder,
                                                    )

    train_dataset, _, _ = prepare_trainer_dataset(tokenizer, 
                                                    config.dataset_path, 
                                                    prefix_mention_is = config.prefix_mention_is,
                                                    evaluate = config.evaluation,
                                                    )
    if config.unlikelihood_loss:
        print('loading trie......')
        with open(config.trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
        print('trie loaded.......')

        trainer = modifiedSeq2SeqTrainer(
                                model=model,                         # the instantiated  Transformers model to be trained
                                args=training_args,                  # training arguments, defined above
                                train_dataset=train_dataset, 
                                fairseq_loss=config.fairseq_loss,  
                                enc_num = config.prompt_tokens_enc,
                                dec_num = config.prompt_tokens_dec,
                                prefix_allowed_tokens_fn = lambda batch_id, sent: trie.get(sent.tolist()),
                                rdrop = config.rdrop,
                            )
    else:
        trainer = modifiedSeq2SeqTrainer(
                                model=model,                         # the instantiated  Transformers model to be trained
                                args=training_args,                  # training arguments, defined above
                                train_dataset=train_dataset, 
                                fairseq_loss=config.fairseq_loss,  
                                enc_num = config.prompt_tokens_enc,
                                dec_num = config.prompt_tokens_dec,
                                rdrop = config.rdrop,
                            )

    trainer.train()
    

def evalu(config):

    from fairseq_beam import SequenceGenerator, PrefixConstrainedBeamSearch, PrefixConstrainedBeamSearchWithSampling

    if config.t5:

        from models import T5EntityPromptModel
        from transformers import T5Tokenizer, T5Config
        from datagen import prepare_trainer_dataset_t5 as prepare_trainer_dataset
        
        t5conf = T5Config.from_pretrained('./t5-large')
        t5conf.dropout_rate = config.dropout

        tokenizer = T5Tokenizer.from_pretrained('./t5-large')

        model = T5EntityPromptModel.from_pretrained(config.model_load_path, 
                                                    config = t5conf,
                                                    n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                    load_prompt = True, 
                                                    soft_prompt_path = config.model_load_path
                                                    )
    
    else:

        from models import BartEntityPromptModel
        from transformers import BartTokenizer, BartConfig
        if config.syn_pretrain:
            from datagen import prepare_trainer_dataset_pre as prepare_trainer_dataset
        else:
            from datagen import prepare_trainer_dataset_fine as prepare_trainer_dataset
        
        tokenizer = BartTokenizer.from_pretrained(config.model_token_path)

        bartconf = BartConfig.from_pretrained(config.model_load_path)
        bartconf.max_position_embeddings = config.max_position_embeddings
        bartconf.attention_dropout = config.attention_dropout
        bartconf.dropout = config.dropout
        bartconf.max_length = config.max_length

        model = BartEntityPromptModel.from_pretrained(config.model_load_path, 
                                                        config = bartconf,
                                                        n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec), 
                                                        load_prompt = True, 
                                                        soft_prompt_path=config.model_load_path,
                                                        )
    
    model = model.cuda().to(model.device)

    _, dev_dataset, test_dataset = prepare_trainer_dataset(tokenizer, 
                                                    config.dataset_path, 
                                                    prefix_mention_is = config.prefix_mention_is,
                                                    evaluate = config.evaluation,
                                                    )

    if config.testset:
        print('eval on test set')
        eval_dataset = test_dataset
    else:
        print('eval on develop set')
        eval_dataset = dev_dataset

    print('loading cui2str dictionary....')
    dict_path = config.dict_path
    if 'json' in dict_path:
        with open(dict_path, 'r') as f:
            cui2str = json.load(f)
    else:
        with open(dict_path, 'rb') as f:
            cui2str = pickle.load(f)

    str2cui = {}
    for cui in cui2str:
        if isinstance(cui2str[cui], list):
            for name in cui2str[cui]:
                if name in str2cui:
                    str2cui[name].append(cui)
                else:
                    str2cui[name] = [cui]
        else:
            name = cui2str[cui]
            if name in str2cui:
                str2cui[name].append(cui)
                print('duplicated vocabulary')
            else:
                str2cui[name] = [cui]
    print('dictionary loaded......')

    if config.rerank:
        print('loading retrieved names......')
        with open(config.retrieved_path, 'r') as f:
            retrieved_names = [line.split('\t')[0].split(' ') for line in f.readlines()]
        print('retrieved names loaded.')
        for i, l in tqdm(enumerate(retrieved_names)):
            for cui in list(l):
                if cui in cui2str:
                    continue
                else:
                    retrieved_names[i].remove(cui)

        print('loading tokenized names......')
        with open(config.dataset_path+'/tokenized.json', 'r') as f:
            tokenized_names = json.load(f)
        print('tokenized names loaded.')
    
    if config.gold_sty:

        print('loading tokenized names......')
        with open(config.dataset_path+'/tokenized.json', 'r') as f:
            tokenized_names = json.load(f)
        print('tokenized names loaded.')

        print('loading sty to cui dict.....')
        with open(config.dataset_path+'/sty2cui.json', 'r') as f:
            sty2cuis = json.load(f)
        with open(config.dataset_path+'/sty.json', 'r') as f:
            cuis2sty = json.load(f)
        print('sty to cui dict loaded.')
        
        trie_dict = {}
        for sty in sty2cuis:
            names = []
            for cui in tqdm(sty2cuis[sty]):
                names += tokenized_names[cui]
            trie_dict[sty] = Trie(names)

    print('loading trie......')
    with open(config.trie_path, "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))
    print('trie loaded.......')

    print('loading label cuis......')
    with open(config.dataset_path+'/testlabel.txt', 'r') as f:
        cui_labels = [set(cui.strip('\n').replace('+', '|').split('|')) for cui in f.readlines()]
    print('label cuis loaded')

    if config.beam_threshold == 0:
        print('without using beam threshold')
        beam_strategy = PrefixConstrainedBeamSearch(
            tgt_dict=None, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
        )
    else:
        beam_strategy = PrefixConstrainedBeamSearchWithSampling(
            tgt_dict=None, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            logit_thresholding=config.beam_threshold,
        )
    
    fairseq_generator = SequenceGenerator(
        models = model,
        tgt_dict = None,
        beam_size=config.num_beams,
        max_len_a=0,
        max_len_b=config.max_length,
        min_len=config.min_length,
        eos=model.config.eos_token_id,
        search_strategy=beam_strategy,

        ##### all hyperparams below are set to default
        normalize_scores=True,
        len_penalty=config.length_penalty,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    )

    results = list()
    cui_results = list()
    results_score = list()

    input_ids = []
    decoder_input_ids = []
    attention_mask = []
    count_top1 = 0
    count_top5 = 0
    for i in tqdm(range(0, len(eval_dataset))):
        
        if config.rerank:
            trie = Trie(sum([tokenized_names[cui] for cui in retrieved_names[i]], []))
            fairseq_generator.search = (PrefixConstrainedBeamSearch(
                                                                    tgt_dict=None, 
                                                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
                                                                    ))
        if config.gold_sty:
            trie = trie_dict[cuis2sty[cui_labels[i]]]
            fairseq_generator.search = (PrefixConstrainedBeamSearch(
                                                                    tgt_dict=None, 
                                                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
                                                                    ))
            
        input_ids.append(test_dataset[i]['input_ids'])
        attention_mask.append(test_dataset[i]['attention_mask'])
        decoder_input_ids.append(test_dataset[i]['decoder_input_ids_test'])

        if i%config.per_device_eval_batch_size == 0:

            input_ids, attention_mask = reform_input(torch.stack(input_ids), attention_mask=torch.stack(attention_mask), ending_token=model.config.eos_token_id)
            sample = {'net_input':{'input_ids':input_ids, 'attention_mask':attention_mask}}
            
            result_tokens, posi_scores = fairseq_generator.forward(
                sample=sample,
                prefix_mention_is = config.prefix_mention_is,
                prefix_tokens=decoder_input_ids[0].unsqueeze(0).cuda() if config.prefix_mention_is else None,
            )

            for ba, beam_sent in enumerate(result_tokens):
                result = []
                cui_result = []
                for be, sent in enumerate(beam_sent):
                    if config.prefix_mention_is:
                        result.append(tokenizer.decode(sent[len(decoder_input_ids[0]):], skip_special_tokens=True))
                    else:
                        result.append(tokenizer.decode(sent, skip_special_tokens=True))
                
                for r in result:
                    if r.strip(' ') in str2cui:
                        cui_result.append(str2cui[r.strip(' ')])
                    else:
                        cui_result.append(r)

                cui_results.append(cui_result)
                results.append(result)
                results_score.append(posi_scores)
                # print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                # print(posi_scores)
                # print(result)
                # print(cui_result)
                # print(cui_labels[i])
                # print(cui2str[cui_labels[i]])
                # input()

                if cui_labels[i].intersection(set(cui_result[0])):
                    count_top1 += 1
                    count_top5 += 1
                elif cui_labels[i].intersection(set(sum(cui_result,[]))):
                    count_top5 += 1

            if i % 50 == 49:
                print('=============Top1 Precision:\t',count_top1/(i+1))
                print('=============Top5 Precision:\t',count_top5/(i+1))

            input_ids = []
            decoder_input_ids = []
            attention_mask = []
    
    print('=============Top1 Precision:\t',count_top1/(i+1))
    print('=============Top5 Precision:\t',count_top5/(i+1))

    with open('./logs.txt', 'a+') as f:
        f.write(str(config.seed) + '======\n')
        f.write(config.model_load_path + '\n')
        f.write('Top1 Precision:\t'+str(count_top1/(i+1))+'\n')
        f.write('Top5 Precision:\t'+str(count_top5/(i+1))+'\n\n')
    
    if config.testset:

        with open(os.path.join(config.model_load_path, 'results_test.pkl'), 'wb') as f:
            pickle.dump([results, results_score], f)

    else:

        with open(os.path.join(config.model_load_path, 'results_dev.pkl'), 'wb') as f:
            pickle.dump([results, results_score], f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training configuration')

    parser.add_argument("dataset_path",type=str,
                        help="path of the medmentions dataset")
    parser.add_argument("-model_save_path",type=str,default='./model_saved',
                        help="path of the pretrained model")
    parser.add_argument("-trie_path",type=str,default='./trie.pkl',
                        help="path of the Trie")
    parser.add_argument("-dict_path",type=str,default='./trie.pkl',
                        help="path of the cui2str dictionary")
    parser.add_argument("-retrieved_path",type=str,default='./trie.pkl',
                        help="path of the cui2str dictionary")
    parser.add_argument("-model_load_path",type=str,default='facebook/bart-large',
                        help="path of the pretrained model")
    parser.add_argument("-model_token_path",type=str,default='facebook/bart-large',
                        help="path of the pretrained model")
    parser.add_argument("-logging_path",type=str, default='./logs',
                        help="path of saved logs")
    parser.add_argument('-logging_steps', type=int, default=500,
                        help='save logs per logging step')                  
    parser.add_argument('-save_steps', type=int, default = 20000,
                        help='save checkpoints per save steps')
    parser.add_argument('-num_train_epochs', type=int, default = 8,
                        help="number of training epochs")
    parser.add_argument('-per_device_train_batch_size', type=int, default = 4,
                        help='training batch size')
    parser.add_argument('-per_device_eval_batch_size', type=int, default = 5,
                        help='evaluation batch size')
    parser.add_argument('-warmup_steps', type=int, default = 500,
                        help='warmup steps')
    parser.add_argument('-finetune', action='store_true',
                        help='if finetune the bart params')
    parser.add_argument('-t5', action='store_true',
                        help='if use t5 pretrained model')
    parser.add_argument('-fairseq_loss', action='store_true',
                        help='if use label smoothed loss in fairseq')
    parser.add_argument('-evaluation', action='store_true',
                        help='whether to train or evaluate')
    parser.add_argument('-testset', action='store_true',
                        help='whether evaluate with testset or devset')
    parser.add_argument('-load_prompt', action='store_true',
                        help='whether to load prompt')
    parser.add_argument('-weight_decay', type=float, default = 0.01,
                        help='weigth decay of optimizer')
    parser.add_argument('-length_penalty', type=float, default = 1,
                        help='length penaltyof beam search')
    parser.add_argument('-beam_threshold', type=float, default = 0,
                        help='logit threshold of beam search')
    parser.add_argument('-unlikelihood_loss', action='store_true',
                        help='whether using unlikelihood loss')
    parser.add_argument('-init_lr', type=float, default = 5e-5,
                        help='initial learning rate of AdamW')
    parser.add_argument('-evaluation_strategy', type=str, default='no',
                        help='evaluation strategy')
    parser.add_argument('-prompt_tokens_enc', type=int, default = 0,
                        help='a tuple containing number of soft prompt tokens in encoder and decoder respectively')
    parser.add_argument('-prompt_tokens_dec', type=int, default = 0,
                        help='a tuple containing number of soft prompt tokens in encoder and decoder respectively')
    parser.add_argument('-seed', type=int, default = 42,
                        help='the seed of huggingface seq2seq training, 42 is also the default of huggingface train default')
    parser.add_argument('-label_smoothing_factor', type=float, default = 0.1,
                        help='label smoothig factor')
    parser.add_argument('-unlikelihood_weight', type=float, default = 0.1,
                        help='label smoothig factor')
    parser.add_argument('-max_grad_norm', type=float, default = 0.1,
                        help='gradient clipping value')
    parser.add_argument('-max_steps', type=int, default = 200000,
                        help='max training steps override num_train_epoch')
    parser.add_argument('-gradient_accumulate', type=int, default = 1,
                        help='max training steps override num_train_epoch')
    parser.add_argument('-lr_scheduler_type', type=str, default = 'polynomial',
                        help='the learning rate schedule type')
    parser.add_argument('-attention_dropout', type=float, default = 0.1,
                        help='the attention dropout')
    parser.add_argument('-dropout', type=float, default = 0.1,
                        help='dropout')
    parser.add_argument('-max_position_embeddings', type=int, default = 1024,
                        help="the max length for position embedding")
    parser.add_argument('-num_beams', type=int, default = 5,
                        help='the attention dropout')
    parser.add_argument('-max_length', type=int, default = 1024,
                        help='the attention dropout')
    parser.add_argument('-min_length', type=int, default = 1,
                        help='the attention dropout')
    parser.add_argument('-sample_train', action='store_true',
                        help='if to use training target sampled by tfidf similarity')
    parser.add_argument('-prefix_prompt', action='store_true',
                        help='wheather use prefix prompt tokens')
    parser.add_argument('-rerank', action='store_true',
                        help='wheather to rerank the retrieved names')
    parser.add_argument('-init_from_vocab', action='store_true',
                        help='wheather initialize prompt from the mean of token embeddings')
    parser.add_argument('-no_finetune_decoder', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-syn_pretrain', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-gold_sty', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-prefix_mention_is', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-rdrop', type=float, default=0.0)



    config = parser.parse_args()

    if config.evaluation:
        evalu(config)
    else:
        train(config)

