import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
from tqdm import tqdm



def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])
        
tfidf_vectorizer = '/media/sda1/GanjinZero/cluster_tfidf/scispacy_based/datasets/tfidf_vectorizer.joblib'
vectorizer = joblib.load(tfidf_vectorizer)

with open('./ncbi_medic_target_kb.pkl', 'rb') as f:
    cui2names = pickle.load(f)

count = 0
chooses = []
res = {}

origin_count = 0
compressed_count = 0
for cui in tqdm(cui2names):
    strs = list(cui2names[cui])
    conv_strs = conv(strs)
    res[cui] = list(set(conv_strs))

# rmv_cnt = 0
# all_x = set()
# repeat_x = set()
# for cui in res:
#     for x in res[cui]:
#         if not x in all_x:
#             all_x.update([x])
#         else:
#             repeat_x.update([x])

# repeat_dict = {x:[] for x in repeat_x}
# for cui in res:
#     for x in res[cui]:
#         if x in repeat_x:
#             repeat_dict[x].append(cui)
# print(len(repeat_dict))

# pop_cnt = 0
# for conv_x in repeat_dict:
#     min_cnt = 100000
#     min_i = None
#     for cui in repeat_dict[conv_x]:
#         if len(res[cui]) < min_cnt:
#             min_cnt = len(res[cui])
#             min_i = cui
#     # print(conv_x, min_cnt, min_i)
#     for cui in repeat_dict[conv_x]:
#         if cui != min_i:
#             res[cui].remove(conv_x)
#             pop_cnt += 1
# print(pop_cnt)
# print(sum([len(res[x]) for x in res]))

# res["C1872584"] = ["Perfluorooctanesulfonate"]
ccc = 0
for key in res:
    print(res[key])
    input()
    if len(res[key]) == 0:
        # print(key)
        ccc += 1
print(ccc)
print(res[key])

import json
with open(f'medic_all_dup_uncased.json', 'w') as f:
    json.dump(res, f, indent=2)
