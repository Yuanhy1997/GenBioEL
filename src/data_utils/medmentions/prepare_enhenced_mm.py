import re
import html
import copy
from tqdm import tqdm
import os
import pickle
import json
from nltk.stem import WordNetLemmatizer

def create_input(doc, max_length=384, start_delimiter="[START_ENT]", end_delimiter="[END_ENT]"):
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
            buffer['annotations'].append({'idx':(int(annotaion[1]), int(annotaion[2])), 'mention':annotaion[3], 'semantic':annotaion[4], 'entity':cui2concept[annotaion[5].strip('UMLS:')].lower(), 'cui':annotaion[5].strip('UMLS:')})
        else:
            if buffer['id'] in train_ids:
                dataset.append(copy.deepcopy(buffer))
    return dataset

def generate_medmentions_files_2017aa(dataset, cui2syn, path):
    
    with open(path+'.source', 'w') as f1, open(path+'.target', 'w') as f2:
        for samples in tqdm(dataset):
            doc = dict()
            for sample in samples['annotations']:
                doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
                doc['meta']['left_context'] = samples['text'][:sample['idx'][0]]
                doc['meta']['right_context'] = samples['text'][sample['idx'][1]:]
                doc['meta']['mention'] = sample['mention']
                doc['input'] = samples['text']
                f1.write(json.dumps([sample['mention'], create_input(doc)]) +'\n')
                f2.write(sample['mention'] + ' is ' + sample['entity']+'\n')
                if 'train' in path:
                    syn_list = cui2syn[sample['cui']]
                    for item in syn_list:
                        doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
                        doc['meta']['left_context'] = samples['text'][:sample['idx'][0]]
                        doc['meta']['right_context'] = samples['text'][sample['idx'][1]:]
                        doc['meta']['mention'] = item
                        doc['input'] = samples['text']
                        f1.write(json.dumps([item, create_input(doc)]) +'\n')
                        f2.write(item + ' is ' + sample['entity']+'\n')

                # f1.write(create_input(doc) +'\n')
                # f2.write(sample['entity']+'\n')
    
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


if __name__ == '__main__':
    path = '/media/sdb1/Hongyi_Yuan/medical_linking/GENRE/2017aa_medmentions_nodup/enhenced_mm_data/'
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/GENRE/2017aa_medmentions_nodup/cui2str_in_mm_select_short.pkl'
    with open(kb_path, 'rb') as f:
        cui2concept = pickle.load(f)
    with open('/media/sdb1/Hongyi_Yuan/medical_linking/GENRE/2017aa_medmentions_nodup/pretrain_data/syn_def_pretrain_in_mm_select_short.json', 'r') as f:
        cui_sets = f.readlines()
    cui2syn = {}
    for i, line in enumerate(cui_sets):
        item = json.loads(line.strip('\n'))
        cui2syn[item[0]] = item[1]['synonyms'] 
    cui2syn = standardize_synonyms(cui2syn)

        
    for part in ['test', 'dev', 'trng']:
        dataset = read_medmentions(cui2concept, part)
        if part == 'trng':
            generate_medmentions_files_2017aa(dataset, cui2syn, path + 'train')
        else:
            generate_medmentions_files_2017aa(dataset, cui2syn,  path + part)

    


