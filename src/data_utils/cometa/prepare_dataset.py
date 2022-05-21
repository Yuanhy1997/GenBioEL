import re
import html
import copy
from tqdm import tqdm
import os
import pickle
import json
from nltk.stem import WordNetLemmatizer
import difflib
import numpy as np
import random

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

def read_cometa(cometa_part = 'train'):
    prefix = '/media/sdb1/Zengsihang/ShenZhen/CODER_faiss_finetune/test/test_sapbert_tasks/evaluation/data/cometa/splits/stratified_general'
    with open(prefix + '/'+cometa_part+'.csv', 'r') as f:
        all_data = f.readlines()[1:]
    
    dataset = []
    buffer = dict()
    count = 0
    for i in range(len(all_data)): 
        sample = all_data[i].strip('\n').split('\t')
        gen_id = sample[3]
        spe_id = sample[5]
        term = sample[1].lower()
        context = sample[6].lower()
        if term in context:
            lr_context = context.split(term, maxsplit=1)
            buffer = {'mention': term, 'left': lr_context[0].strip(' '), 'right':lr_context[1].strip(' '), 'gen_id':gen_id, 'spe_id':spe_id}
        else:
            count += 1
            buffer = {'mention': term, 'left': '', 'right':context.strip(' '), 'gen_id':gen_id, 'spe_id':spe_id}
        dataset.append(copy.deepcopy(buffer))
        # if '|t|' in all_data[i]:
        #     buffer['text'] = all_data[i].strip('\n').split('|', maxsplit=2)[-1]
        #     buffer['id'] = all_data[i].strip('\n').split('|', maxsplit=2)[0]
        # elif '|a|' in all_data[i]:
        #     buffer['text'] += ' '
        #     buffer['text'] += all_data[i].strip('\n').split('|', maxsplit=2)[-1]
        #     buffer['annotations'] = list()
        # elif buffer['id'] in all_data[i]:
        #     annotaion = all_data[i].strip('\n').split('\t')
        #     if '|' in annotaion[5]:
        #         code = [item.strip('OMIM:') for item in annotaion[5].split('|')]
        #     elif '+' in annotaion[5]:
        #         code = [item.strip('OMIM:') for item in annotaion[5].split('+')]
        #     else:
        #         code = [annotaion[5].strip('OMIM:')]
        #     buffer['annotations'].append({'idx':(int(annotaion[1]), int(annotaion[2])), 'mention':annotaion[3], 'semantic':annotaion[4], 'cui':code})
        # else:
        #     dataset.append(copy.deepcopy(buffer))
    print(count / len(all_data))
    return dataset

def prepare_input_for_ab3p(dataset, path):
    with open('../../Ab3P/input_'+path+'.txt', 'w') as f:
        for samples in tqdm(dataset):
            f.write(samples['id'] + '| ' + samples['text'] +'\n')

def read_ab3p_result(path):
    with open('./ab3p_out/'+path+'.out', 'r') as f:
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
    return sim[0].argmax(), np.max(np.array(sim)[0])
# def generate_cometa_training_multi_ab3p(dataset, path, cui2multiname, ab3p_result):
    
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
def sample_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    sim = np.exp(np.array(sim)[0] + 0.005)
    prob = sim/np.sum(sim)
    return np.random.choice(a = len(prob), size = 30, p = prob, replace = True)


def generate_cometa_training_multi_ab3p_tfidf_sample(dataset, path, cui2multiname, id_type, vectorizer):
    
    with open(path+'.source', 'w') as f1:
        for sample in tqdm(dataset):
            doc = dict()
            mention = sample['mention']#.lower()
            doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
            doc['meta']['left_context'] = sample['left']
            doc['meta']['right_context'] = sample['right']
            doc['meta']['mention'] = mention
            doc['input'] = sample['left'] + ' ' + sample['mention'] + ' ' + sample['right']
            if len(cui2multiname[sample[id_type]]) == 1:
                f1.write(json.dumps([mention, create_input(doc)]) +'\n')
            else:
                f1.write(json.dumps([mention, create_input(doc)]) +'\n')

    for sample in tqdm(dataset):
        doc = dict()
        mention = sample['mention']#.lower()
        if len(cui2multiname[sample[id_type]]) != 1:
            sample['sampled_idx'] = sample_similarity_tfidf(cui2multiname[sample[id_type]], mention, vectorizer)

    with open(path+'.target', 'w') as f2:
        for k in tqdm(range(30)):
            for sample in tqdm(dataset):
                mention = sample['mention']#.lower()
                if len(cui2multiname[sample[id_type]]) == 1:
                    sss = json.dumps([mention + ' is', cui2multiname[sample[id_type]][0]])
                    f2.write(sss+'\n')
                else:
                    idx = sample['sampled_idx'][k]
                    # idx = sample_similarity_tfidf(cui2multiname[sample['cui']], mention.lower(), vectorizer)
                    sss = json.dumps([mention + ' is', cui2multiname[sample[id_type]][idx]])
                    f2.write(sss+'\n')

def generate_cometa_training_multi_ab3p_tfidf(dataset, path, cui2multiname, id_type, vectorizer):
    count = set()
    count_mention = 0

    tfidfscore = []
    with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
        for sample in tqdm(dataset):
            doc = dict()
            if sample[id_type] in cui2multiname:
                name_set = cui2multiname[sample[id_type]]
                doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
                doc['meta']['left_context'] = sample['left']
                doc['meta']['right_context'] = sample['right']
                doc['meta']['mention'] = sample['mention']
                doc['input'] = sample['left'] + ' ' + sample['mention'] + ' ' + sample['right']
                # if len(name_set) == 1:
                #     f1.write(json.dumps([create_input(doc)]) +'\n')
                #     f2.write(json.dumps([sample['mention'] + ' is', name_set[0]])+'\n')
                # else:
                #     # idx = cal_similarity_tfidf(name_set, sample['mention'], vectorizer)
                #     idx = random.randint(0, len(name_set)-1)
                #     f1.write(json.dumps([create_input(doc)]) +'\n')
                #     f2.write(json.dumps([sample['mention'] + ' is', name_set[idx]])+'\n')

                if len(name_set) == 1:
                    f1.write(json.dumps([create_input(doc).replace('START ', '').replace('END ', '')]) +'\n')
                    f2.write(json.dumps([create_input(doc).split('START')[0], name_set[0]])+'\n')
                else:
                    idx, mmmm = cal_similarity_tfidf(name_set, sample['mention'], vectorizer)
                    tfidfscore.append(mmmm)
                    f1.write(json.dumps([create_input(doc).replace('START ', '').replace('END ', '')]) +'\n')
                    f2.write(json.dumps([create_input(doc).split('START')[0], name_set[idx]])+'\n')
            else:
                # count.update([sample['cui']])
                count_mention += 1
    with open('./tfidfscore.pkl', 'wb') as f:
        pickle.dump(tfidfscore, f)
    input()
    print(path, len(count), count_mention)

def generate_cui_label(dataset, path, cui2multiname, id_type):
    with open(path + 'label.txt', 'w') as f:
        for sample in tqdm(dataset):
            if sample[id_type] in cui2multiname:
                f.write(sample[id_type]+'\n')
        



if __name__ == '__main__':


    path = './cometa_gen_bart/'

    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/COMETA/cometa_gen/snomedct_final.json'
    with open(kb_path, 'r') as f:
        cui2multiname = json.load(f)
    for cui in cui2multiname:
        cui2multiname[cui] = sorted(cui2multiname[cui], key = lambda i: len(i))
    # for cui in copy.deepcopy(list(cui2multiname.keys())):
    #     # if not cui2multiname[cui]:
    #     #     cui2multiname.pop(cui)
    #     cui2multiname[cui] = [cui2multiname[cui]]
    
    # dataset = read_cometa('test')
    # generate_cui_label(dataset, path, cui2multiname, 'gen_id')


    tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
    vectorizer = joblib.load(tfidf_vectorizer)

    mention_cui = set()
    for part in ['test', 'dev', 'train']:
        dataset = read_cometa(part)
    #     for item in dataset:
    #         mention_cui.update([item['gen_id']])
    # with open('./mention_cuis', 'wb') as f:
    #     pickle.dump(mention_cui, f)
        generate_cui_label(dataset, path + part, cui2multiname, 'gen_id')
        generate_cometa_training_multi_ab3p_tfidf(dataset, path + part, cui2multiname, 'gen_id', vectorizer)
    # for part in ['train']:
    #     dataset = read_cometa(part)
    #     generate_cometa_training_multi_ab3p_tfidf_sample(dataset, path + part, cui2multiname, 'gen_id', vectorizer)

    