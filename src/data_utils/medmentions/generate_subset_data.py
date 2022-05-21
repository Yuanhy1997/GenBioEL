
import json
from tqdm import tqdm
import pickle
#'COMETA/cometa_gen'
#'BC5CDR/bc5cdr'
with open('../BC5CDR/bc5cdr/train.target', 'r') as f, open('../BC5CDR/bc5cdr/trainlabel.txt', 'r') as g:
    train_mentions = [json.loads(item.strip('\n'))[0].strip(' is') for item in f.readlines()]
    train_cui = [cui.strip('\n') for cui in g.readlines()]
    train_cui_set = set(train_cui)
    train_mentions_set = set(train_mentions)
with open('../BC5CDR/bc5cdr/test.target', 'r') as f, open('../BC5CDR/bc5cdr/testlabel.txt', 'r') as g:
    test_mentions = [json.loads(item.strip('\n'))[0].strip(' is') for item in f.readlines()]
    test_cui = [cui.strip('\n') for cui in g.readlines()]
    test_cui_set = set(test_cui)
    test_mentions_set = set(test_mentions)
with open('../BC5CDR/bc5cdr/mesh_final.json', 'r') as f:
    cui2str = json.load(f)

str2cui = {}
for cui in cui2str:
    if isinstance(cui2str[cui], list):
        for name in cui2str[cui]:
            if name in str2cui:
                str2cui[name].append(cui)
            else:
                str2cui[name] = [cui]
    else:
        name = cui2str[cui]
        if name in str2cui:
            str2cui[name].append(cui)
            print('duplicated vocabulary')
        else:
            str2cui[name] = [cui]

with open('/media/sdb1/Hongyi_Yuan/bc5cdr_bart.json', 'r') as f:
    results = json.load(f)
results_cui = [[str2cui[s.strip(' ')] for s in item] for item in results]
print(len(results_cui))
unseen_cui_samples = []
for i, cui in tqdm(enumerate(test_cui)):
    if cui not in train_cui_set:
        unseen_cui_samples.append(i)
print(len(unseen_cui_samples))
count = len(unseen_cui_samples)
count1 = 0
count5 = 0
for i in unseen_cui_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100) 
# with open('./bc5cdr_cui_zeroshot.pkl', 'wb') as f:
#     pickle.dump(unseen_cui_samples, f)


unseen_mention_samples = []
for i, mention in tqdm(enumerate(test_mentions)):
    if mention not in train_mentions_set:
        unseen_mention_samples.append(i)
print(len(unseen_mention_samples))
count = len(unseen_mention_samples)
count1 = 0
count5 = 0
for i in unseen_mention_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100) 
# with open('./bc5cdr_mention_zeroshot.pkl', 'wb') as f:
#     pickle.dump(unseen_cui_samples, f)
# print('done')
# input()
# input()

multispan_mention_samples = []
unispan_mention_samples = []
for i, mention in tqdm(enumerate(test_mentions)):
    if len(mention.split(' ')) == 1:
        unispan_mention_samples.append(i)
    else:
        multispan_mention_samples.append(i)
print(len(multispan_mention_samples))
print(len(unispan_mention_samples))
count = len(multispan_mention_samples)
count1 = 0
count5 = 0
for i in multispan_mention_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100) 
count = len(unispan_mention_samples)
count1 = 0
count5 = 0
for i in unispan_mention_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100) 


notmatch_mention_samples = []
for i, (cui, mention) in tqdm(enumerate(zip(test_cui, test_mentions))):
    if mention not in cui2str[cui]:
        notmatch_mention_samples.append(i)
print(len(notmatch_mention_samples))
count = len(notmatch_mention_samples)
count1 = 0
count5 = 0
for i in notmatch_mention_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100)


cui_count = {cui:0 for cui in train_cui_set}
for cui in train_cui:
    cui_count[cui] += 1
top100_cui = [result[0] for result in sorted(cui_count.items(), key=lambda k:-k[1])[:100]]
top100_cui_samples = []
for i, cui in tqdm(enumerate(test_cui)):
    if cui in top100_cui:
        top100_cui_samples.append(i)
print(len(top100_cui_samples))
count = len(top100_cui_samples)
count1 = 0
count5 = 0
for i in top100_cui_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100)


mention2cui = {mention:{} for mention in train_mentions}
for i, (cui, mention) in enumerate(zip(train_cui, train_mentions)):
    if cui in mention2cui[mention]:
        mention2cui[mention][cui] += 1
    else:
        mention2cui[mention][cui] = 1
unpopular_cui = {}
print(len(mention2cui))
print(len(train_mentions))
for mention in mention2cui:
    # if len(mention2cui[mention]) > 1:
        # print(mention2cui[mention])
        # input()
    if len(mention2cui[mention])>1:
        for cui in sorted(mention2cui[mention].items(), key=lambda k:k[1]):
            if mention in unpopular_cui:
                unpopular_cui[mention].append(cui[0])
            else:
                unpopular_cui[mention] = [cui[0]]
print(len(unpopular_cui))
unpopular_cui_samples = []
for i, (cui, men) in tqdm(enumerate(zip(test_cui, test_mentions))):
    if men in unpopular_cui:# and cui in unpopular_cui[men]:
        unpopular_cui_samples.append(i)
print(len(unpopular_cui_samples))
count = len(unpopular_cui_samples)
count1 = 0
count5 = 0
for i in unpopular_cui_samples:
    if test_cui[i] in results_cui[i][0]:
        count1 += 1
    if test_cui[i] in sum(results_cui[i],[]):
        count5 += 1
print(count1/count*100, count5/count*100)

