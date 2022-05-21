import re
import html
import copy
from tqdm import tqdm
import os
import pickle
import json
from nltk.stem import WordNetLemmatizer
import random
import numpy as np

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = np.array(features_b.dot(features_T).todense())
    return sim[0]+0.1

def standardize_synonyms(cui2syn):
    wnl = WordNetLemmatizer()
    for cui in tqdm(cui2syn):
        syn_list = cui2syn[cui]
        std_syn_list = []
        clean_syn_list = []
        for syn in syn_list:
            std_syn = ' '.join([wnl.lemmatize(s) for s in syn.lower().split(' ')])
            if std_syn not in std_syn_list:
                std_syn_list.append(std_syn)
                clean_syn_list.append(syn)
        cui2syn[cui] = clean_syn_list
    return cui2syn

def standardize_defs(cui2def):
    from googletrans import Translator
    translator = Translator(service_urls=['translate.google.cn'])
    for cui in tqdm(cui2def):
        def_list = cui2def[cui]
        clean_def_list = []
        for des in def_list:
            if translator.detect(des).lang == 'en':
                clean_def_list.append(des)
            else:
                clean_def_list.append(translator.translate(des).text)
        cui2def[cui] = clean_def_list
    return cui2def

template_sets = ['is defined as', 'is described as', 'are the definations of', 'describe', 'define']
def prepare_origin_pretraindata(cui2concept, cui2defs, cui2syns, special_tokens = None, template = False):
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    cui2syns = standardize_synonyms(cui2syns)
    # cui2defs = standardize_defs(cui2defs)
    output = {}
    for cui in tqdm(cui2defs):
        context = []
        for syn in cui2syns[cui]:
            random.shuffle(cui2defs[cui])
            if template:
                idx = random.randint(0, 4)
                if idx < 2:
                    if not special_tokens:
                        des = ' '.join([syn, template_sets[idx]] + cui2defs[cui])
                    else:
                        des = ' '.join([special_tokens[0], syn, special_tokens[1], template_sets[idx]] + cui2defs[cui])
                    tks = tokenizer(des)['input_ids']
                    if len(tks) > 700:
                        print('truncate!!', len(tks), ' to 800')
                        des = tokenizer.decode(tks[:700])
                else:
                    if not special_tokens:
                        des = ' '.join(cui2defs[cui] + [template_sets[idx], syn])
                    else:
                        des = ' '.join(cui2defs[cui] + [template_sets[idx], special_tokens[0], syn, special_tokens[1]])
                    tks = tokenizer(des)['input_ids']
                    if len(tks) > 700:
                        print('truncate!!', len(tks), ' to 800')
                        des = tokenizer.decode(tks[-700:])
            else:
                des = ' '.join(cui2defs[cui])
                tks = tokenizer(des)['input_ids']
                if len(tks) > 700:
                    print('truncate!!', len(tks), ' to 800')
                    des = tokenizer.decode(tks[:700])

            context.append([syn, des])
        output[cui] = [cui2concept[cui], context]
    return output


def generate_pair(y, mentions, vectorizer, select_scheme):
    if select_scheme == 'random':
        return random.choice(mentions)
    elif select_scheme == 'sample':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        # print(similarity_estimate.shape)
        return np.random.choice(mentions, 1, p = similarity_estimate/np.sum(similarity_estimate))[0]
    elif select_scheme == 'most_sim':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        return mentions[similarity_estimate.argmax()]
    elif select_scheme == 'least_sim':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        return mentions[similarity_estimate.argmin()]
    else:
        print('Wrong mention selection scheme input!!!')

template_sets = ['is defined as', 'is described as', 'are the definations of', 'describe', 'define']
def prepare_final_pretraindata(cui2defs, cui2syns, special_tokens = None, template = False, select_scheme = 'random'):
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    cui2syns = standardize_synonyms(cui2syns)
    # cui2defs = standardize_defs(cui2defs)
    output = []
    for cui in tqdm(cui2defs):
        for syn in cui2syns[cui]:
            if len(cui2syns[cui]) > 2:
                pending_set = copy.deepcopy(cui2syns[cui])
                pending_set.remove(syn)
                mention = generate_pair(syn, pending_set, vectorizer, select_scheme)
            elif len(cui2syns[cui]) == 2:
                pending_set = copy.deepcopy(cui2syns[cui])
                pending_set.remove(syn)
                mention = pending_set[0]
            else:
                mention = syn

            random.shuffle(cui2defs[cui])

            if template:
                idx = random.randint(0, 4)
                if idx < 2:
                    if not special_tokens:
                        des = ' '.join([mention, template_sets[idx]] + cui2defs[cui][:2])
                    else:
                        des = ' '.join([special_tokens[0], mention, special_tokens[1], template_sets[idx]] + cui2defs[cui][:2])
                    
                    tks = tokenizer(des)['input_ids']
                    if len(tks) > 700:
                        print('truncate!!', len(tks), ' to 800')
                        if not special_tokens:
                            des = ' '.join([mention, template_sets[idx]] + cui2defs[cui][:1])
                        else:
                            des = ' '.join([special_tokens[0], mention, special_tokens[1], template_sets[idx]] + cui2defs[cui][:1])
                else:
                    if not special_tokens:
                        des = ' '.join(cui2defs[cui][:2] + [template_sets[idx], mention])
                    else:
                        des = ' '.join(cui2defs[cui][:2] + [template_sets[idx], special_tokens[0], mention, special_tokens[1]])
                    
                    tks = tokenizer(des)['input_ids']
                    if len(tks) > 700:
                        print('truncate!!', len(tks), ' to 800')
                        if not special_tokens:
                            des = ' '.join(cui2defs[cui][:1] + [template_sets[idx], mention])
                        else:
                            des = ' '.join(cui2defs[cui][:1] + [template_sets[idx], special_tokens[0], mention, special_tokens[1]])
            else:
                des = ' '.join(cui2defs[cui][:2])
                tks = tokenizer(des)['input_ids']
                if len(tks) > 700:
                    print('truncate!!', len(tks), ' to 800')
                    des = ' '.join(cui2defs[cui][:1])

            if cui2defs[cui]:
                output.append([cui, syn, mention, des])
            else:
                output.append([cui, syn, mention, ''])

    return output
                


if __name__ == '__main__':
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/MedMentions/2017aa_mm_final/cui2str_small_multi_all_uncased.pkl'
    with open(kb_path, 'rb') as f:
        cui2syns = pickle.load(f)
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/GENRE/2017aa_medmentions_nodup/pretrain_data/syn_def_pretrain_in_mm_select_short.json'
    cui2defs = {}
    with open(kb_path, 'r') as f:
        data = f.readlines()
        for line in data:
            l = json.loads(line.strip('\n'))
            cui2defs[l[0]] = l[1]['def']
    for cui in cui2defs:
        for i in range(len(cui2defs[cui])):
            cui2defs[cui][i] = cui2defs[cui][i].lower()
        for i in range(len(cui2syns[cui])):
            cui2syns[cui][i] = cui2syns[cui][i].lower()
    
    tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
    vectorizer = joblib.load(tfidf_vectorizer)
    
    special_tokens = ["[START_ENT]", "[END_ENT]"]
    template = True
    select_scheme = 'random'
    data = prepare_final_pretraindata(cui2defs, cui2syns, special_tokens, template, select_scheme)

    with open('./2017aa_mm_final/pretrain_data/final_pretrain_random/data.json', 'w') as f:
        for item in tqdm(data):
            s = json.dumps(item)
            f.write(s+'\n')

    

        
    


