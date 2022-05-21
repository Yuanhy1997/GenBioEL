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

# import json
# with open('./bc5cdr/mesh_final.json', 'r') as f:
#     cui2names = json.load(f)

# # result_cui2name = {}
# # for cui in cui2names:
# #     for name in cui2names[cui]:
# #         if cui not in result_cui2name:
# #             result_cui2name[cui] = name
# #         elif len(name) < len(result_cui2name[cui]):
# #             result_cui2name[cui] = name
# result_cui2name = {}
# for cui in cui2names:
#     if cui2names[cui]:
#         result_cui2name[cui] = random.choice(cui2names[cui])

# with open(f'mesh_final_random.json', 'w') as f:
#     json.dump(result_cui2name, f, indent=2)
# print('done')
# input()

def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    # x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])
        
tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
vectorizer = joblib.load(tfidf_vectorizer)

with open('./bc5cdr_mesh_target_kb.pkl', 'rb') as f:
    cui2names = pickle.load(f)

with open('./mention_cdr', 'rb') as f:
    mention_cdr = pickle.load(f)
print(len(mention_cdr))


count = 0
chooses = []
res = {}

origin_count = 0
compressed_count = 0
for cui in tqdm(cui2names):
    strs = list(cui2names[cui])
    conv_strs = conv(strs)
    res[cui] = list(set(conv_strs))


name2cui = {}
for cui in res:
    for name in res[cui]:
        if name in name2cui:
            count += 1
            # print(name)
            # print(res[name2cui[name]])
            # print(res[cui])
            # input()
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
        if cui != min_i:
            res[cui].remove(conv_x)
            pop_cnt += 1
print(pop_cnt)
print(sum([len(res[x]) for x in res]))

check_dup(res)
count_no_name_cui(res, mention_cdr)

import json
with open(f'mesh_final_cased.json', 'w') as f:
    json.dump(res, f, indent=2)
