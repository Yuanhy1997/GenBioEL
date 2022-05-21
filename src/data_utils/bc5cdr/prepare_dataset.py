import re
import html
import copy
from tqdm import tqdm
import os
import pickle
import json
from nltk.stem import WordNetLemmatizer
import difflib
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

def read_bc5cdr(bc5cdr_part = 'train'):
    
    with open('/media/sdb1/Hongyi_Yuan/medical_linking/BC5CDR/data/'+bc5cdr_part+'.txt', 'r') as f:
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
            annotation = all_data[i].strip('\n').split('\t')
            if len(annotation) == 6:
                buffer['annotations'].append({'idx':(int(annotation[1]), int(annotation[2])), 'mention':annotation[3], 'semantic':annotation[4], 'cui':annotation[5]})
            if len(annotation) == 7 and '|' not in annotation[-2]:
                buffer['annotations'].append({'idx':(int(annotation[1]), int(annotation[2])), 'mention':annotation[3], 'semantic':annotation[4], 'cui':annotation[5]})
            if len(annotation) == 7 and '|' in annotation[-2]:
                if ' '.join(annotation[6].split('|')) == annotation[3]:
                    offset = 0
                    for cui, mention in zip(annotation[5].split('|'), annotation[6].split('|')):
                        buffer['annotations'].append({'idx':(int(annotation[1])+offset, int(annotation[1])+offset+len(mention)), 'mention':mention, 'semantic':annotation[4], 'cui':cui})
                        offset += 1+len(mention)
                elif len(annotation[-1].split('|')) == 2 and ' and ' in annotation[3]:
                    # if ', ' not in annotation[3]:
                    offset = 0
                    for mention, cui in zip(annotation[3].split(' and '), annotation[5].split('|')):
                        buffer['annotations'].append({'idx':(int(annotation[1])+offset, int(annotation[1])+offset+len(mention)), 'mention':mention, 'semantic':annotation[4], 'cui':cui})
                        offset += 5+len(mention)
                    # else:
                    #     offset = 0
                    #     for mention, cui in zip(annotation[3].replace(' and ', ', ').split(', '), annotation[5].split('|')):
                    #         buffer['annotations'].append({'idx':(int(annotation[1])+offset, int(annotation[1])+len(mention)), 'mention':mention, 'semantic':annotation[4], 'cui':cui})
                    #         offset += 2+len(mention)
                    #     buffer['annotations'][-1]['idx'] = (int(annotation[2])-len(mention), int(annotation[2]))
                elif len(annotation[-1].split('|')) == 2 and ' or ' in annotation[3]:
                    offset = 0
                    for mention, cui in zip(annotation[3].split(' or '), annotation[5].split('|')):
                        buffer['annotations'].append({'idx':(int(annotation[1])+offset, int(annotation[1])+offset+len(mention)), 'mention':mention, 'semantic':annotation[4], 'cui':cui})
                        offset += 4+len(mention)
                # elif len(annotation[-1].split('|')) == 2 and annotation[-1].split('|')[1] in annotation[-1].split('|')[0]:
                #     cuis = annotation[5].split('|')
                #     buffer['annotations'].append({'idx':(int(annotation[1]), int(annotation[2])), 'mention':annotation[3], 'semantic':annotation[4], 'cui':cuis[0]})
                #     mentions = annotation[-1].split('|')[-1]
                #     buffer['annotations'].append({'idx':(int(annotation[2])-len(mentions), int(annotation[2])), 'mention':mentions, 'semantic':annotation[4], 'cui':cuis[1]})

                # else:
                #     print(annotation)


        else:
            dataset.append(copy.deepcopy(buffer))

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
# def generate_bc5cdr_training_multi_ab3p(dataset, path, cui2multiname, ab3p_result):
    
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
import numpy as np
def sample_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    print(sim)
    sim = np.exp(np.array(sim)[0] + 0.005)
    prob = sim/np.sum(sim)
    print(prob)
    input()
    return np.random.choice(a = len(prob), size = 30, p = prob, replace = True)


def generate_bc5cdr_training_multi_ab3p_tfidf(dataset, path, cui2multiname, ab3p_result, vectorizer):
    count = set()
    count_mention = 0
    f3 = open(path+'label.txt', 'w')
    with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
        for samples in tqdm(dataset):
            doc = dict()
            for sample in samples['annotations']:
                if sample['cui'] in cui2multiname:
                    if samples['id'] in ab3p_result and sample['mention'] in ab3p_result[samples['id']]:
                        mention = ab3p_result[samples['id']][sample['mention']].lower()
                    else:
                        mention = sample['mention'].lower()
                    doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
                    doc['meta']['left_context'] = samples['text'][:sample['idx'][0]].lower()
                    doc['meta']['right_context'] = samples['text'][sample['idx'][1]:].lower()
                    doc['meta']['mention'] = mention.lower()
                    doc['input'] = samples['text']
                    if len(cui2multiname[sample['cui']]) == 1:
                        f1.write(json.dumps([create_input(doc)]) +'\n')
                        f2.write(json.dumps([mention + ' is', cui2multiname[sample['cui']][0]])+'\n')
                        count_mention += 1
                    else:
                        idx = cal_similarity_tfidf(cui2multiname[sample['cui']], mention, vectorizer)
                        f1.write(json.dumps([create_input(doc)]) +'\n')
                        f2.write(json.dumps([mention + ' is', cui2multiname[sample['cui']][idx]])+'\n')
                        count_mention += 1
                else:
                    count.update([sample['cui']])
    if f3:
        f3.close()


if __name__ == '__main__':

    path = './bc5cdr_bart/'

    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/BC5CDR/bc5cdr/mesh_final.json'
    with open(kb_path, 'r') as f:
        cui2multiname = json.load(f)
    for cui in list(cui2multiname.keys()):
        cui2multiname[cui] = cui2multiname[cui]
    for cui in cui2multiname:
        cui2multiname[cui] = sorted(cui2multiname[cui], key = lambda i: len(i))

    tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
    vectorizer = joblib.load(tfidf_vectorizer)
    for part in ['test', 'dev','train']:
        dataset = read_bc5cdr(part)
        ab3p_result = read_ab3p_result(part)
        generate_bc5cdr_training_multi_ab3p_tfidf(dataset, path + part, cui2multiname, ab3p_result, vectorizer)
    # mention_cdr = set()
    # for part in ['test','dev','train']:
    #     dataset = read_bc5cdr(part)
    #     for abstract in dataset:
    #         for item in abstract['annotations']:
    #             mention_cdr.update([item['cui']])
    # with open('./mention_cdr', 'wb') as f:
    #     pickle.dump(mention_cdr, f)
        # ab3p_result = read_ab3p_result(part)
        # generate_bc5cdr_training_multi_ab3p_tfidf_sample(dataset, path + part, cui2multiname, ab3p_result, vectorizer)

    