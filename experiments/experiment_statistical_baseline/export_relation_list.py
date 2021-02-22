""" convert data from jsonl to csv """
import json
import csv
import pandas as pd
from itertools import chain

data = ['sat', 'u2', 'u4', 'google', 'bats']


# for relation embedding model training
for i in data:
    all_pair = []
    for t in ['test.jsonl', 'valid.jsonl']:
        with open('./data/{}/{}'.format(i, t), 'r') as f:
            jsons = list(map(lambda x: json.loads(x), f.read().split('\n')))
        stem = list(map(lambda x: x['stem'], jsons))
        choice = list(chain(*map(lambda x: x['choice'], jsons)))
        all_pair += stem + choice
    tmp = pd.DataFrame(all_pair, columns=None)
    tmp.to_csv('pairs_{}.csv'.format(i).format(), sep="\t", quoting=csv.QUOTE_NONE, index=False, header=False)

