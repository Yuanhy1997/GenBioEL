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
def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])

def load_target_dict(input_files):
    if 'json' in input_files:
        with open(input_files, 'r') as f:
            cui2str = json.load(f)
        return cui2str
        
    with open(input_files, 'r') as f:
        lines = f.readlines()
    cui2str = {}
    for l in lines:
        cui, name = l.strip('\n').split('||')
        cui = cui.replace('+', '|').split('|')
        for c in cui:
            if c in cui2str:
                cui2str[c].append(name)
            else:
                cui2str[c] = [name]
                
    # with open('./ncbi_final.json', 'w') as f:
    #     json.dump(cui2str, f)

    return cui2str
        

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

def cal_similarity(a, b):
    sim_list = []
    for item in a:
        sim_list.append(difflib.SequenceMatcher(None, item, b).ratio())
    return sim_list

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = np.array(features_b.dot(features_T).todense())[0]
    # sim_diff = np.array(cal_similarity(a, b))
    # print(a)
    # print(b)
    # print(sim+0.1*sim_diff)
    # input()
    return sim.argmax(), np.max(np.array(sim))

def read_biosyn_outs(part):
    path = '/media/sdb1/Hongyi_Yuan/medical_linking/NCBI/BioSyn/preprocess/ncbi-disease_2'
    files = os.listdir(os.path.join(path, 'processed_'+part))
    dataset = {}
    for f in files:
        pmid = f.strip('.concept')
        with open(path+'/'+part + '/' + pmid + '.txt', 'r') as f:
            lines = [l.strip('\n') for l in f.readlines()]
        context = ' '.join([l for l in lines if l]).lower()
        with open(path+'/processed_'+part + '/' + pmid + '.concept', 'r') as f:
            lines = [l.strip('\n') for l in f.readlines()]
        annotations = [l.split('||') for l in lines]
        dataset[pmid] = {'text': context, 'annotation': annotations}
    return dataset

def generate_files(dataset, part, vectorizer, cui2names, remove_composite):
    f1 = open('./ncbi/'+part+'.source', 'a')
    f2 = open('./ncbi/'+part+'.target', 'a')
    f3 = open('./ncbi/'+part+'label.txt', 'a')
    count = 0
    tfidfscore = []
    for pmid in tqdm(dataset):
        doc = dict()
        for ant in dataset[pmid]['annotation']:
            lidx, ridx = ant[1].split('|')
            mention = ant[3]
            cuis = ant[4].strip().replace('+', '|').split('|')
            if len(cuis) > 1 and remove_composite:
                count += 1
                continue
            pending_name = []
            for cui in cuis:
                pending_name += cui2names[cui]
            for ment in mention.split('|'):
                doc['meta'] = {"left_context":None, "mention":None, "right_context":None}
                doc['meta']['left_context'] = dataset[pmid]['text'][:int(lidx)].lower()
                doc['meta']['right_context'] = dataset[pmid]['text'][int(ridx):].lower()
                doc['meta']['mention'] = ment.lower()
                doc['input'] = dataset[pmid]['text']
                if len(pending_name) == 1:
                    f1.write(json.dumps([create_input(doc)]) +'\n')
                    f2.write(json.dumps([ment.lower() + ' is', pending_name[0]])+'\n')
                else:
                    idx, mmmm = cal_similarity_tfidf(pending_name, ment.lower(), vectorizer)
                    f1.write(json.dumps([create_input(doc)]) +'\n')
                    f2.write(json.dumps([ment.lower() + ' is', pending_name[idx]])+'\n')
                f3.write(ant[4].strip().replace('+', '|')+'\n')
                if 'test' or 'dev' in part:
                    break
    f1.close()
    f2.close()
    print(count)
    f3.close()

if __name__ == '__main__':


    # for part in ['test', 'dev', 'train']:
    #     dataset = read_ncbi(part)
    #     prepare_input_for_ab3p(dataset, part)
    # input()

    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/NCBI/BioSyn/preprocess/ncbi-disease_2/test_dictionary.txt'
    cui2multiname = load_target_dict(kb_path)
    with open('./ncbi_mesh_target_kb.pkl', 'rb') as f:
        mesh2str = pickle.load(f)
    for cui in mesh2str:
        if 'MSHU' in cui:
            mesh2str[cui.strip('MSHU')] = mesh2str[cui]
    count = 0
    for cui in cui2multiname:
        if cui in mesh2str:
            cui2multiname[cui] = list(set(conv(cui2multiname[cui] + mesh2str[cui])))
            count += len(cui2multiname[cui])
    for cui in copy.deepcopy(list(cui2multiname.keys())):
        if not cui2multiname[cui]:
            cui2multiname.pop(cui)
    with open('./ncbi/ncbi_final.json', 'w') as f:
        json.dump(cui2multiname, f)
    print(count, len(cui2multiname))
    

    tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
    vectorizer = joblib.load(tfidf_vectorizer)

    for part in ['test']:
        dataset = read_biosyn_outs(part)
        generate_files(dataset, part, vectorizer, cui2multiname, remove_composite=False)
    
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/NCBI/BioSyn/preprocess/ncbi-disease_2/dev_dictionary_fordata.txt'
    cui2multiname = load_target_dict(kb_path)
    with open('./ncbi_mesh_target_kb.pkl', 'rb') as f:
        mesh2str = pickle.load(f)
    for cui in mesh2str:
        if 'MSHU' in cui:
            mesh2str[cui.strip('MSHU')] = mesh2str[cui]
    count = 0
    for cui in cui2multiname:
        if cui in mesh2str:
            cui2multiname[cui] = list(set(conv(cui2multiname[cui] + mesh2str[cui])))
            count += len(cui2multiname[cui])
    for cui in copy.deepcopy(list(cui2multiname.keys())):
        if not cui2multiname[cui]:
            cui2multiname.pop(cui)
    for cui in cui2multiname:
        cui2multiname[cui] = sorted(cui2multiname[cui], key = lambda i: len(i))
    import copy
    cui2multiname_dev = copy.deepcopy(cui2multiname)

    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/NCBI/BioSyn/preprocess/ncbi-disease_2/train_dictionary.txt'
    cui2multiname = load_target_dict(kb_path)
    with open('./ncbi_mesh_target_kb.pkl', 'rb') as f:
        mesh2str = pickle.load(f)
    for cui in mesh2str:
        if 'MSHU' in cui:
            mesh2str[cui.strip('MSHU')] = mesh2str[cui]
    count = 0
    for cui in cui2multiname:
        if cui in mesh2str:
            cui2multiname[cui] = list(set(conv(cui2multiname[cui] + mesh2str[cui])))
            count += len(cui2multiname[cui])
    for cui in copy.deepcopy(list(cui2multiname.keys())):
        if not cui2multiname[cui]:
            cui2multiname.pop(cui)
    for cui in cui2multiname:
        cui2multiname[cui] = sorted(cui2multiname[cui], key = lambda i: len(i))
    import copy
    cui2multiname_train = copy.deepcopy(cui2multiname)

    for part in ['train', 'dev']:
        if part == 'train':
            dataset = read_biosyn_outs('train')
            generate_files(dataset, part, vectorizer, cui2multiname_train, remove_composite=True)
            dataset = read_biosyn_outs('dev')
            generate_files(dataset, part, vectorizer, cui2multiname_dev, remove_composite=True)
        else:
            dataset = read_biosyn_outs(part)
            generate_files(dataset, part, vectorizer, cui2multiname_dev, remove_composite=False)

    