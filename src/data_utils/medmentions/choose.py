import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
from tqdm import tqdm
import json
import ipdb
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
# import random
# with open('./medmentions/st21pv_final.json', 'r') as f:
#     cui2names = json.load(f)

# print(cui2names['C0085262'])
# input()
# input()
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

# with open(f'st21pv_final_random.json', 'w') as f:
#     json.dump(result_cui2name, f, indent=2)
# print('done')
# input()

def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])
        
tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
vectorizer = joblib.load(tfidf_vectorizer)

with open('./st21pv_all_kb_cased.json', 'rb') as f:
    cui2names = json.load(f)

with open('./st21pv_all_kb_sty.json', 'rb') as f:
    cui2stys = json.load(f)

sty2cuis = {}
for cui in tqdm(cui2stys):
    if cui2stys[cui] in sty2cuis:
        sty2cuis[cui2stys[cui]].append(cui)
    else:
        sty2cuis[cui2stys[cui]] = [cui]
print(len(sty2cuis))
# with open('./st21pv_all_sty2cui.json', 'w') as f:
#     json.dump(sty2cuis, f)
# print('done')

input()

import pickle
with open('/media/sdb1/Hongyi_Yuan/medical_linking/MedMentions/2017aa_mm_final_old/cui2str_small_multi_uncased.pkl', 'rb') as f:
    inmm = pickle.load(f)
print(len(inmm))

# cui2names.pop('C3252361')

count = 0
chooses = []
res = {}
origin_count = 0
compressed_count = 0
for cui in tqdm(cui2names):
    strs = list(cui2names[cui])
    # strs = [s+' '+cui2stys[cui] for s in strs]
    conv_strs = conv(strs)
    res[cui] = list(set(conv_strs))
print(sum([len(res[x]) for x in res]))

name_count = {}
for cui in tqdm(res):
    for n in res[cui]:
        if n in name_count:
            name_count[n] += 1
        else:
            name_count[n] = 1
cccc = 0
for cui in tqdm(res):
    for i in range(len(res[cui])):
        if name_count[res[cui][i]] > 1:
            cccc += 1
            res[cui][i] = res[cui][i] + ' ' + cui2stys[cui]
print(cccc)


cui2names = res     
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

check_dup(res)
count_no_name_cui(res, inmm)

# import json
# with open(f'st21pv_final_sufsty.json', 'w') as f:
#     json.dump(res, f, indent=2)



# name_count_dict = {}
# for cui in res:
#     for item in res[cui]:
#         if item in name_count_dict:
#             name_count_dict[item] += 1
#         else:
#             name_count_dict[item] = 1
# cui2len = {}
# for cui in res:
#     cui2len[cui] = sum([1/name_count_dict[name] for name in res[cui]])
# sorted_cuis = [i[0] for i in sorted(cui2len.items(), key = lambda kv:(kv[1], kv[0]))]

# for cui in res:
#     res[cui] = sorted(res[cui], key = lambda kv:name_count_dict[kv])

# result = {cui:[] for cui in res}
# for cui in tqdm(sorted_cuis):
#     for name in res[cui]:
#         if name_count_dict[name] == 1:
#             result[cui].append(name)
#     res[cui] = list(set(res[cui]) - set(result[cui]))
#     if len(res[cui]) == 0:
#         a = res.pop(cui)

# check_dup(result)
# # input()

# print(len(res))
# used_names = set()
# for cui in tqdm(res):
#     used_names.update(res[cui])

# used_names = list(used_names)    

# for cui in tqdm(res):
#     if res[cui]:
#         if not result[cui] and res[cui]:
#             for name in res[cui]:
#                 if name in used_names:
#                     used_names.remove(name)
#                     result[cui].append(name)
#                     break
#             res[cui].remove(name)
# check_dup(result)
# while used_names:
#     for cui in tqdm(res):
#         if res[cui]:
#             for name in res[cui]:
#                 if name in used_names:
#                     used_names.remove(name)
#                     result[cui].append(name)
#                     break
#             res[cui].remove(name)
#     check_dup(result)
#     # input() 

    
# res = result
