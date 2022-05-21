import re
import html
import copy
from tqdm import tqdm
import os
import pickle
import json
from nltk.stem import WordNetLemmatizer
import difflib
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    return sim[0].argmax()


def conv(x):

    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])

def dedup_dict(cui2names):
    count = 0
    chooses = []
    res = {}
    origin_count = 0
    compressed_count = 0
    for cui in tqdm(cui2names):
        strs = list(cui2names[cui])
        conv_strs = conv(strs)
        res[cui] = list(set(conv_strs))
    print(sum([len(res[x]) for x in res]))
    name2cui = {}
    for cui in res:
        for name in res[cui]:
            if name in name2cui:
                count += 1
            name2cui[name] = cui
    print(count)
    rmv_cnt = 0
    all_x = set()
    repeat_x = set()
    for cui in res:
        for x in res[cui]:
            if not x in all_x:
                all_x.update([x])
            else:
                repeat_x.update([x])

    repeat_dict = {x:[] for x in repeat_x}
    for cui in res:
        for x in res[cui]:
            if x in repeat_x:
                repeat_dict[x].append(cui)
    print(len(repeat_dict))
    pop_cnt = 0
    for conv_x in repeat_dict:
        min_cnt = 100000
        min_i = None
        for cui in repeat_dict[conv_x]:
            if len(res[cui]) < min_cnt:
                min_cnt = len(res[cui])
                min_i = cui
        # print(conv_x, min_cnt, min_i)
        for cui in repeat_dict[conv_x]:
            if cui != min_i and len(res[cui])>1:
                res[cui].remove(conv_x)
                pop_cnt += 1
    print(pop_cnt)
    print(sum([len(res[x]) for x in res]))
    return res

def load_target_dict(input_files):

    with open(input_files, 'r') as f:
        cui2str_raw = json.load(f)

    cui2str = {}
    for cui in cui2str_raw:
        name_set = []
        for l in cui2str_raw[cui]:
            name_set += l[1:]
        name_set = conv(name_set)
        cui2str[cui] = name_set

    return cui2str

special_tokens = ['START', 'END']
def generate_files(part, cui2names, fold, vectorizer, dataset):
    prefix = '/media/sdb1/Hongyi_Yuan/medical_linking/AskPatients/askpatients'
    if not os.path.exists(prefix + '/fold'+str(fold)):
        os.mkdir(prefix + '/fold'+str(fold))
    
    nameset = set()
    for cui in cui2names:
        nameset.update(cui2names[cui])

    f1 = open(prefix + '/fold'+str(fold)+'/'+part+'.source', 'w')
    f2 = open(prefix + '/fold'+str(fold)+'/'+part+'.target', 'w')
    f3 = open(prefix + '/fold'+str(fold)+'/'+part+'label.txt', 'w')
    for data in tqdm(dataset):
        cui, nam, text = data.strip('\n').split('\t')
        for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
            text = text.replace(ch, ' ')
        pending_name = cui2names[cui]
        if len(pending_name) == 1:
            f1.write(json.dumps([special_tokens[0]+' '+text.lower()+' '+special_tokens[1]]) +'\n')
            f2.write(json.dumps([text.lower() + ' is', pending_name[0]])+'\n')
        else:
            # print(pending_name, text)
            idx = cal_similarity_tfidf(pending_name, text.lower(), vectorizer)
            f1.write(json.dumps([special_tokens[0]+' '+text.lower()+' '+special_tokens[1]]) +'\n')
            f2.write(json.dumps([text.lower() + ' is', pending_name[idx]])+'\n')
        f3.write(cui.strip()+'\n')
        if 'train' in part and nam.lower() in nameset:
            # print('aaas')
            f1.write(json.dumps([special_tokens[0]+' '+text.lower()+' '+special_tokens[1]]) +'\n')
            f2.write(json.dumps([text.lower() + ' is', nam.lower()])+'\n')
            f3.write(cui.strip()+'\n')

    f1.close()
    f2.close()
    f3.close()

def clean_aap_dataset(dataset):
    result_dataset = []
    former_code = ''
    for data in tqdm(dataset):
        code, tag, text = data.strip('\n').split('\t')
        if code == former_code and text == tag:
            continue
        result_dataset.append(data)
        former_code = code
    print('resulted cleaned dataset:', len(result_dataset))
    return result_dataset



if __name__ == '__main__':

    snomed_ct = '/media/sdb1/Hongyi_Yuan/medical_linking/COMETA/snomedct_uncased_dup.json'
    with open(snomed_ct, 'r') as f:
        snomed_ct2str = json.load(f)
    # snomed_ct2str = {}
    
    kb_path = '/media/sdb1/Hongyi_Yuan/medical_linking/rescnn_bioel-main/resources/ontologies/askapatient.json'
    cui2multiname = load_target_dict(kb_path)
    count = 0
    for cui in copy.deepcopy(list(cui2multiname.keys())):
        if not cui2multiname[cui]:
            cui2multiname.pop(cui)
        if cui in snomed_ct2str:
            buffer = list(set(cui2multiname[cui]+snomed_ct2str[cui]))
            count += (len(buffer)-len(cui2multiname[cui]))
            cui2multiname[cui] = buffer
    cui2multiname = dedup_dict(cui2multiname)
    for cui in cui2multiname:
        cui2multiname[cui] = sorted(cui2multiname[cui], key = lambda i: len(i))
    with open('./askpatients_final.json', 'w') as f:
        json.dump(cui2multiname, f)
    
    tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
    vectorizer = joblib.load(tfidf_vectorizer)

    for fold in range(10):
        for part in ['test', 'dev', 'train']:
            if part == 'train':
                with open('/media/sdb1/Hongyi_Yuan/medical_linking/AskPatients/AskAPatient/AskAPatient.fold-'+str(fold)+'.' + 'train' + '.txt', 'r', encoding='ISO-8859-1') as f:
                    dataset = f.readlines()
                dataset = clean_aap_dataset(dataset)
                generate_files(part, cui2multiname, fold, vectorizer, dataset)
            elif part =='dev':
                with open('/media/sdb1/Hongyi_Yuan/medical_linking/AskPatients/AskAPatient/AskAPatient.fold-'+str(fold)+'.' + 'validation' + '.txt', 'r', encoding='ISO-8859-1') as f:
                    dataset = f.readlines()
                generate_files(part, cui2multiname, fold, vectorizer, dataset)
            else:
                with open('/media/sdb1/Hongyi_Yuan/medical_linking/AskPatients/AskAPatient/AskAPatient.fold-'+str(fold)+'.' + 'test' + '.txt', 'r', encoding='ISO-8859-1') as f:
                    dataset = f.readlines()
                generate_files(part, cui2multiname, fold, vectorizer, dataset)

    