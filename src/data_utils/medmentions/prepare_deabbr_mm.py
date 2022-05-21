import re
import html
import copy
from tqdm import tqdm
import os
import pickle
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import difflib
import numpy as np


def create_input_short(doc, max_length=384, start_delimiter="START", end_delimiter="END"):
    if "meta" in doc and all(
        e in doc["meta"] for e in ("left_context", "mention", "right_context")
    ):
        doc["meta"]["left_context"] = sent_tokenize(doc["meta"]["left_context"].strip(' '))
        doc["meta"]["right_context"] = sent_tokenize(doc["meta"]["right_context"].strip(' '))
        if len(doc["meta"]["left_context"]) == 0 or (len(doc["meta"]["left_context"]) < 2 and doc["meta"]["left_context"][-1][-1] != '.'):
            input_ = (
                ' '.join(doc["meta"]["left_context"])
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + ' '.join(doc["meta"]["right_context"][:3])
            )
        elif len(doc["meta"]["right_context"]) < 2:
            input_ = (
                ' '.join(doc["meta"]["left_context"][-3:])
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + ' '.join(doc["meta"]["right_context"])
            )
        else:
            input_ = (
                ' '.join(doc["meta"]["left_context"][-2:])
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + ' '.join(doc["meta"]["right_context"][:2])
            )

    input_ = html.unescape(input_.strip(' '))

    return input_


def create_input(doc, max_length=384, start_delimiter="START", end_delimiter="END"):
    if "meta" in doc and all(
        e in doc["meta"] for e in ("left_context", "mention", "right_context")
    ):
        doc["meta"]["left_context"] = doc["meta"]["left_context"].strip(' ')
        doc["meta"]["right_context"] = doc["meta"]["right_context"].strip(' ')

        if len(doc["input"].split(" ")) <= max_length:
            input_ = (
                doc["meta"]["left_context"]
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + doc["meta"]["right_context"]
            )
        elif len(doc["meta"]["left_context"].split(" ")) <= max_length // 2:
            input_ = (
                doc["meta"]["left_context"]
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + " ".join(
                    doc["meta"]["right_context"].split(" ")[
                        : max_length - len(doc["meta"]["left_context"].split(" "))
                    ]
                )
            )
        elif len(doc["meta"]["right_context"].split(" ")) <= max_length // 2:
            input_ = (
                " ".join(
                    doc["meta"]["left_context"].split(" ")[
                        len(doc["meta"]["right_context"].split(" ")) - max_length :
                    ]
                )
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + doc["meta"]["right_context"]
            )
        else:
            input_ = (
                " ".join(doc["meta"]["left_context"].split(" ")[-max_length // 2 :])
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + " ".join(doc["meta"]["right_context"].split(" ")[: max_length // 2])
            )
    else:
        input_ = doc["input"]

    input_ = html.unescape(input_.strip(' '))

    return input_

def read_medmentions(cui2concept, medmention_part = 'trng'):
    
    with open('./full/data/corpus_pubtator_pmids_'+medmention_part+'.txt', 'r') as f:
        train_ids = f.readlines()
    for i in range(len(train_ids)):
        train_ids[i] = train_ids[i].strip('\n')

    with open('./st21pv/data/corpus_pubtator.txt', 'r') as f:
        all_data = f.readlines()
    
    dataset = []
    buffer = dict()
    for i in range(len(all_data)): 
        if '|t|' in all_data[i]:
            buffer['text'] = all_data[i].strip('\n').split('|', maxsplit=2)[-1]
            buffer['id'] = all_data[i].strip('\n').split('|', maxsplit=2)[0]
        elif '|a|' in all_data[i]:
            buffer['text'] += ' '
            buffer['text'] += all_data[i].strip('\n').split('|', maxsplit=2)[-1]
            buffer['annotations'] = list()
        elif buffer['id'] in all_data[i]:
            annotaion = all_data[i].strip('\n').split('\t')
            # if annotaion[5].strip('UMLS:') in cui2concept:
            buffer['annotations'].append({'idx':(int(annotaion[1]), int(annotaion[2])), 'mention':annotaion[3], 'semantic':annotaion[4], 'cui':annotaion[5].strip('UMLS:')})
        else:
            if buffer['id'] in train_ids:
                dataset.append(copy.deepcopy(buffer))
    return dataset


# def generate_medmentions_training_lower(dataset, path):
    
#     with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
#         for samples in tqdm(dataset):
#             doc = dict()
#             for sample in samples['annotations']:
#                 doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
#                 doc['meta']['left_context'] = samples['text'][:sample['idx'][0]].lower()
#                 doc['meta']['right_context'] = samples['text'][sample['idx'][1]:].lower()
#                 doc['meta']['mention'] = sample['mention'].lower()
#                 doc['input'] = samples['text']
#                 f1.write(json.dumps([sample['mention'].lower(), create_input(doc)]) +'\n')
#                 f2.write(sample['mention'].lower() + ' is ' + sample['entity']+'\n')

# def generate_medmentions_training_origin(dataset, path):
    
#     with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
#         for samples in tqdm(dataset):
#             doc = dict()
#             for sample in samples['annotations']:
#                 doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
#                 doc['meta']['left_context'] = samples['text'][:sample['idx'][0]]
#                 doc['meta']['right_context'] = samples['text'][sample['idx'][1]:]
#                 doc['meta']['mention'] = sample['mention']
#                 doc['input'] = samples['text']
#                 f1.write(json.dumps([sample['mention'], create_input(doc)]) +'\n')
#                 f2.write(sample['mention'] + ' is ' + sample['entity']+'\n')


def prepare_input_for_ab3p(dataset, path):
    with open('../../Ab3P/input_'+path+'.txt', 'w') as f:
        for samples in tqdm(dataset):
            f.write(samples['id'] + '| ' + samples['text'] +'\n')


def read_ab3p_result(path):
    with open('./deabbr/deabbr_'+path+'.txt', 'r') as f:
        output = f.readlines()
    deabbr_result = {} # id to dict
    for line in tqdm(output):
        if line.startswith('Failed to find sentence'):
            continue
        if line.split('|', maxsplit = 1)[0].isnumeric():
            id_ = line.split('|', maxsplit = 1)[0]
        else:
            l = line.strip(' ').split('|')
            if id_ in deabbr_result:
                deabbr_result[id_][l[0]] = l[1]
            else:
                deabbr_result[id_] = {l[0]: l[1]}
    return deabbr_result
        
    
# def generate_medmentions_training_ab3p(dataset, path, ab3p_result):
    
#     with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
#         for samples in tqdm(dataset):
#             doc = dict()
#             for sample in samples['annotations']:
#                 if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
#                     mention = ab3p_result[samples['id']][sample['mention']]
#                 else:
#                     mention = sample['mention']
#                 doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
#                 doc['meta']['left_context'] = samples['text'][:sample['idx'][0]]
#                 doc['meta']['right_context'] = samples['text'][sample['idx'][1]:]
#                 doc['meta']['mention'] = mention
#                 doc['input'] = samples['text']
#                 f1.write(json.dumps([mention, create_input(doc)]) +'\n')
#                 f2.write(mention + ' is ' + sample['entity']+'\n')

def cal_similarity(a, b):
    sim_list = []
    for item in a:
        sim_list.append(difflib.SequenceMatcher(None, item, b).ratio())
    return sim_list.index(max(sim_list))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    return sim[0].argmax()



# def generate_medmentions_training_multi(dataset, path, cui2multiname):
    
#     with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
#         for samples in tqdm(dataset):
#             doc = dict()
#             for sample in samples['annotations']:
#                 doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
#                 doc['meta']['left_context'] = samples['text'][:sample['idx'][0]]
#                 doc['meta']['right_context'] = samples['text'][sample['idx'][1]:]
#                 doc['meta']['mention'] = sample['mention']
#                 doc['input'] = samples['text']
#                 if len(cui2multiname[sample['cui']]) == 1:
#                     f1.write(json.dumps([sample['mention'], create_input(doc)]) +'\n')
#                     f2.write(sample['mention'] + ' is ' + cui2multiname[sample['cui']][0]+'\n')
#                 else:
#                     idx = cal_similarity(cui2multiname[sample['cui']], sample['mention'].lower())
#                     f1.write(json.dumps([sample['mention'], create_input(doc)]) +'\n')
#                     f2.write(sample['mention'] + ' is ' + cui2multiname[sample['cui']][idx]+'\n')

# def generate_medmentions_training_multi_ab3p(dataset, path, cui2multiname, ab3p_result):
    
#     with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
#         for samples in tqdm(dataset):
#             doc = dict()
#             for sample in samples['annotations']:
#                 if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
#                     mention = ab3p_result[samples['id']][sample['mention']]
#                 else:
#                     mention = sample['mention']
#                 doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
#                 doc['meta']['left_context'] = samples['text'][:sample['idx'][0]].lower()
#                 doc['meta']['right_context'] = samples['text'][sample['idx'][1]:].lower()
#                 doc['meta']['mention'] = mention.lower()
#                 doc['input'] = samples['text']
#                 if len(cui2multiname[sample['cui']]) == 1:
#                     f1.write(json.dumps([mention.lower(), create_input(doc)]) +'\n')
#                     f2.write(mention.lower() + ' is ' + cui2multiname[sample['cui']][0]+'\n')
#                 else:
#                     idx = cal_similarity(cui2multiname[sample['cui']], mention.lower())
#                     f1.write(json.dumps([mention.lower(), create_input(doc)]) +'\n')
#                     f2.write(mention.lower() + ' is ' + cui2multiname[sample['cui']][idx]+'\n')
# import random
def generate_medmentions_training_multi_ab3p_tfidf(dataset, path, cui2multiname, ab3p_result, vectorizer):
    
    with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
        for samples in tqdm(dataset):
            doc = dict()
            for sample in samples['annotations']:
                if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
                    mention = ab3p_result[samples['id']][sample['mention']]
                else:
                    mention = sample['mention']
                doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
                doc['meta']['left_context'] = samples['text'][:sample['idx'][0]].lower()
                doc['meta']['right_context'] = samples['text'][sample['idx'][1]:].lower()
                doc['meta']['mention'] = mention.lower()
                doc['input'] = samples['text']
                if len(cui2multiname[sample['cui']]) == 1:
                    f1.write(json.dumps([create_input_short(doc)]) +'\n')
                    f2.write(json.dumps([mention.lower() + ' is', cui2multiname[sample['cui']][0]])+'\n')
                else:
                    idx = cal_similarity_tfidf(cui2multiname[sample['cui']], mention.lower(), vectorizer)
                    # idx = random.randint(0, len(cui2multiname[sample['cui']])-1)
                    f1.write(json.dumps([create_input_short(doc)]) +'\n')
                    f2.write(json.dumps([mention.lower() + ' is', cui2multiname[sample['cui']][idx]])+'\n')


def sample_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    sim = np.exp(np.array(sim)[0] + 0.005)
    prob = sim/np.sum(sim)
    return np.random.choice(a = len(prob), size = 15, p = prob, replace = True)


# def generate_medmentions_training_multi_ab3p_tfidf_sample(dataset, path, cui2multiname, ab3p_result, vectorizer):
    
#     with open(path+'.source', 'w') as f1:
#         for samples in tqdm(dataset):
#             doc = dict()
#             for sample in samples['annotations']:
#                 if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
#                     mention = ab3p_result[samples['id']][sample['mention']]
#                 else:
#                     mention = sample['mention']
#                 doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
#                 doc['meta']['left_context'] = samples['text'][:sample['idx'][0]].lower()
#                 doc['meta']['right_context'] = samples['text'][sample['idx'][1]:].lower()
#                 doc['meta']['mention'] = mention.lower()
#                 doc['input'] = samples['text']
#                 if len(cui2multiname[sample['cui']]) == 1:
#                     f1.write(json.dumps([mention.lower(), create_input(doc)]) +'\n')
#                 else:
#                     f1.write(json.dumps([mention.lower(), create_input(doc)]) +'\n')

#     for samples in tqdm(dataset):
#         doc = dict()
#         for sample in samples['annotations']:
#             if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
#                 mention = ab3p_result[samples['id']][sample['mention']]
#             else:
#                 mention = sample['mention']
#             if len(cui2multiname[sample['cui']]) != 1:
#                 sample['sampled_idx'] = sample_similarity_tfidf(cui2multiname[sample['cui']], mention.lower(), vectorizer)



#     with open(path+'.target', 'w') as f2:
#         for k in tqdm(range(15)):
#             for samples in tqdm(dataset):
#                 doc = dict()
#                 for sample in samples['annotations']:
#                     if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
#                         mention = ab3p_result[samples['id']][sample['mention']]
#                     else:
#                         mention = sample['mention']
#                     if len(cui2multiname[sample['cui']]) == 1:
#                         sss = json.dumps([mention.lower() + ' is', cui2multiname[sample['cui']][0]])
#                         f2.write(sss+'\n')
#                     else:
#                         idx = sample['sampled_idx'][k]
#                         # idx = sample_similarity_tfidf(cui2multiname[sample['cui']], mention.lower(), vectorizer)
#                         sss = json.dumps([mention.lower() + ' is', cui2multiname[sample['cui']][idx]])
#                         f2.write(sss+'\n')


def generate_cui_label(dataset, path):
    with open(path + 'label.txt', 'w') as f:
        for samples in tqdm(dataset):
            for sample in samples['annotations']:
                f.write(sample['cui']+'\n')

if __name__ == '__main__':
    path = './medmentions_sufsty_part_shortcontext/'
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/MedMentions/2017aa_mm_final_old/cui2str_small_short_uncased.pkl'
    with open(kb_path, 'rb') as f:
        cui2concept = pickle.load(f)
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/MedMentions/medmentions_sufsty_part/st21pv_final_sufsty_part.json'
    with open(kb_path, 'rb') as f:
        cui2multiname = json.load(f)
    for cui in cui2multiname:
        if isinstance(cui2multiname[cui], list):
            cui2multiname[cui] = list(set(cui2multiname[cui]))
        else:
            cui2multiname[cui] = [cui2multiname[cui]]
    for cui in cui2multiname:
        cui2multiname[cui] = sorted(cui2multiname[cui], key = lambda i: len(i))

    
    # dataset = read_medmentions(cui2concept, 'test')
    # generate_cui_label(dataset, path, cui2multiname)
    # input()
    # with open('/media/sdb1/Hongyi_Yuan/medical_linking/GENRE/2017aa_medmentions_nodup/pretrain_data/syn_def_pretrain_in_mm_select_short.json', 'r') as f:
    #     cui_sets = f.readlines()
    # cui2syn = {}
    # for i, line in enumerate(cui_sets):
    #     item = json.loads(line.strip('\n'))
    #     cui2syn[item[0]] = item[1]['synonyms'] 
    # cui2syn = standardize_synonyms(cui2syn)\

    tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
    vectorizer = joblib.load(tfidf_vectorizer)

    for part in ['test', 'dev', 'trng']:
        dataset = read_medmentions(cui2concept, part)
        if part == 'trng':
            ab3p_result = read_ab3p_result('train')
            generate_cui_label(dataset, path + part)
            generate_medmentions_training_multi_ab3p_tfidf(dataset, path + 'train', cui2multiname, ab3p_result, vectorizer)
        else:
            ab3p_result = read_ab3p_result(part)
            generate_cui_label(dataset, path + part)
            generate_medmentions_training_multi_ab3p_tfidf(dataset, path + part, cui2multiname, ab3p_result, vectorizer)

    # for part in ['trng']:
    #     dataset = read_medmentions(cui2concept, part)
    #     if part == 'trng':
    #         ab3p_result = read_ab3p_result('train')
    #         generate_medmentions_training_multi_ab3p_tfidf_sample(dataset, path + 'train', cui2multiname, ab3p_result, vectorizer)
        # else:
        #     ab3p_result = read_ab3p_result(part)
        #     generate_medmentions_training_multi_ab3p_tfidf(dataset, path + part, cui2multiname, ab3p_result, vectorizer)
          

    