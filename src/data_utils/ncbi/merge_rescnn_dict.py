import json

with open('../rescnn_bioel-main/resources/ncbi-disease/ncbi-disease/dev_dictionary.txt', 'r') as f:
    data = f.readlines()
with open('../rescnn_bioel-main/resources/ncbi-disease/ncbi-disease/test_dictionary.txt', 'r') as f:
    data += f.readlines()
with open('../rescnn_bioel-main/resources/ncbi-disease/ncbi-disease/train_dictionary.txt', 'r') as f:
    data += f.readlines()

ncbi_dict = {}
for line in data:
    line = line.strip('\n').split('||')
    for cui in line[0].split('|'):
        if cui not in ncbi_dict:
            ncbi_dict[cui] = [line[1].lower()]
        else:
            ncbi_dict[cui].append(line[1].lower())
        # if line[0] not in ncbi_dict:
        #     ncbi_dict[line[0]] = [line[1].lower()]
        # else:
        #     ncbi_dict[line[0]].append(line[1].lower())
print(len(ncbi_dict))
for code in ncbi_dict:
    ncbi_dict[code] = list(set(ncbi_dict[code]))


# print(len(ncbi_dict))
# import pickle
# with open('./ncbi_medic_target_kb.pkl', 'wb') as f:
#     pickle.dump(ncbi_dict, f)

# count = set()
# for code in ncbi_dict:
#     for c in code.split('|'):
#         count.update([c])
# print(len(count))
# print(list(count)[:1000])


import pandas as pd

medic = pd.read_csv('./CTD_diseases.csv', header=None)

medic_kb = {}
for code, name, syns in zip(medic[1], medic[0], medic[7]):
    code = code.strip('MESH:').strip('OMIM:')
    if isinstance(syns, str):
        medic_kb[code] = syns.lower().split('|') + [name.lower()]
    else:
        medic_kb[code] = [name.lower()]
    medic_kb[code] = list(set(medic_kb[code]))
print(len(medic_kb))
import pickle
with open('./medic_origin_target_kb.pkl', 'wb') as f:
    pickle.dump(ncbi_dict, f)
