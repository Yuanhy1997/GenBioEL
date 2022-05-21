import sys
from tqdm import tqdm
import pickle
import pandas as pd
sys.path.append("./")
import torch
import json


# with open('./2017aa_mm_final/cui2str_small_short_uncased.pkl', 'rb') as f:
#     cui2tokens = pickle.load(f)

# print(len(cui2tokens.values())-len(set(cui2tokens.values())))
# for c in cui2tokens:
#     print(cui2tokens[c])
#     input()


def get_minlength_term(tokenizer, term_list):
    length = []
    for token in term_list:
        if not token.isupper():
            length.append(len(tokenizer(' ' + token.lower())['input_ids']))
        else:
            length.append(1e5)
    if min(length) == 1e5:
        length = []
        for token in term_list:
            length.append(len(tokenizer(' ' + token)['input_ids']))
    return term_list[length.index(min(length))], min(length)

# number_of_syn = 0
# number_of_syn_2 = 0
# with open('./2017aa_medmentions_nodup/syn_def_pretrain_in_mm_select_short.json', 'w') as f:
#     for cui in tqdm(UMLS.cui2content):
#         if cui in cui2str_in_mm:
#             UMLS.cui2content[cui]['synonyms'] = list(set(UMLS.cui2content[cui]['synonyms']))
#             UMLS.cui2content[cui]['def'] = list(set(UMLS.cui2content[cui]['def']))
#             line = json.dumps([cui, UMLS.cui2content[cui]], ensure_ascii=False)
#             f.write(line+'\n')
            # number_of_syn += len(UMLS.cui2content[cui]['synonyms'])
            # if len(UMLS.cui2content[cui]['synonyms']) > :
            #     number_of_syn_2 += len(UMLS.cui2content[cui]['synonyms'])
# print(number_of_syn/len(UMLS.cui2content), number_of_syn_2/len(UMLS.cui2content))

with open('/media/sda1/GanjinZero/cluster_tfidf/choose_syn/choose_in_mm_all.json', 'r') as f:
    cui2strlist = json.load(f)
for cui in cui2strlist:
    cui2strlist[cui] = [name.lower() for name in cui2strlist[cui]]
with open('./2017aa_mm_final/cui2str_small_multi_all_uncased.pkl', 'wb') as f:
    pickle.dump(cui2strlist, f)

# from transformers import BartTokenizer
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# cui2tokens = {}
# token2cuis = {}
# for cui in tqdm(cui2strlist):
#     if not cui2strlist[cui]:
#         print(cui)
#         print(cui2strlist[cui])
#     term, _ = get_minlength_term(tokenizer, cui2strlist[cui])
#     cui2tokens[cui] = term
#     token2cuis[term] = cui
# with open('./2017aa_mm_final/cui2str_small_short_uncased.pkl', 'wb') as f:
#     pickle.dump(cui2tokens, f)

    

