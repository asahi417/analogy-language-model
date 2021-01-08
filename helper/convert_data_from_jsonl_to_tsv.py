""" convert data from jsonl to tsv for relation embedding model training """

import json
import csv
from itertools import chain

import pandas as pd

data = ['data/sat_package_v3.jsonl', 'data/u2.jsonl', 'data/u4.jsonl']

# for relation embedding model training
for i in data:
    with open(i, 'r') as f:
        jsons = list(map(lambda x: json.loads(x), f.read().split('\n')))
        stem = list(map(lambda x: x['stem'], jsons))
        choice = list(chain(*map(lambda x: x['choice'], jsons)))
        all_pair = stem + choice
        tmp = pd.DataFrame(all_pair, columns=None)
        # tmp.to_csv(i.replace('.jsonl', '.pair.csv'), index=False, header=False)
        tmp.to_csv(i.replace('.jsonl', '.pair.txt'), sep="\t", quoting=csv.QUOTE_NONE, index=False, header=False)