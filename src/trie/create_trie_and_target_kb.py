import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import pickle
import pandas as pd
from transformers import BartTokenizer
from trie import Trie

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

with open('../benchmark/bc5cdr/target_kb.json', 'rb') as f:
    cui2str = pickle.load(f)

entities = []
for cui in cui2str:
    entities += cui2str[cui]
trie = Trie([16]+list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in tqdm(entities)).trie_dict
with open('../benchmark/bc5cdr/trie.pkl', 'wb') as w_f:
    pickle.dump(trie, w_f)
print("finish running!")
