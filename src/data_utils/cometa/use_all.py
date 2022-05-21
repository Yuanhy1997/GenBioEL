import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
from tqdm import tqdm
import random


def check_dup(res):
    name2cui = {}
    count = 0
    for cui in res:
        for name in res[cui]:
            if name in name2cui:
                count += 1
            name2cui[name] = cui
    print(count)

def count_no_name_cui(res, mm):
    ccc = 0
    bbb = 0
    for key in res:
        if len(res[key]) == 0 and key in mm:
            print(key)
            ccc += 1
        if len(res[key]) == 0:
            bbb += 1
    print(ccc, bbb)

import json
with open('./cometa_gen/snomedct_final.json', 'r') as f:
    cui2names = json.load(f)

result_cui2name = {}
for cui in cui2names:
    for name in cui2names[cui]:
        if cui not in result_cui2name:
            result_cui2name[cui] = name
        elif len(name) < len(result_cui2name[cui]):
            result_cui2name[cui] = name
# result_cui2name = {}
# for cui in cui2names:
#     if cui2names[cui]:
#         result_cui2name[cui] = random.choice(cui2names[cui])

with open(f'snomedct_final_short.json', 'w') as f:
    json.dump(result_cui2name, f, indent=2)
print('done')
input()

def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    # x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])
        
tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
vectorizer = joblib.load(tfidf_vectorizer)

with open('./snomed_target_kb.pkl', 'rb') as f:
    cui2names = pickle.load(f)
import json
with open('/media/sdb1/Hongyi_Yuan/medical_linking/rescnn_bioel-main/resources/ontologies/snomedct.json', 'r') as f:
    snomed2names = json.load(f)
for code in snomed2names:
    snomed2names[code] = set(sum([l[1:] for l in snomed2names[code]], []))
    if code in cui2names:
        snomed2names[code].update(cui2names[code])
    
cui2names = snomed2names
with open('./mention_cuis', 'rb') as f:
    incometa = pickle.load(f)
print(len(incometa))
count = 0
chooses = []
res = {}

origin_count = 0
compressed_count = 0
for cui in tqdm(cui2names):
    strs = list(cui2names[cui])
    conv_strs = conv(strs)
    res[cui] = list(set(conv_strs))


rmv_cnt = 0
all_x = set()
repeat_x = set()
repeat_cui = set()
for cui in res:
    for x in res[cui]:
        if not x in all_x:
            all_x.update([x])
        else:
            repeat_x.update([x])
            repeat_cui.update([cui])
print(len(repeat_cui))

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
            if min_cnt == 1:
                break
    # print(conv_x, min_cnt, min_i)
    for cui in repeat_dict[conv_x]:
        if cui != min_i:
            res[cui].remove(conv_x)
            pop_cnt += 1
print(pop_cnt)
print(sum([len(res[x]) for x in res]))

# res["C1872584"] = ["Perfluorooctanesulfonate"]

check_dup(res)
count_no_name_cui(res, incometa)

import json
with open(f'snomedct_final_cased.json', 'w') as f:
    json.dump(res, f, indent=2)
