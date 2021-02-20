import os
import alm
from random import randint, seed

import pandas as pd

seed(1234)
os.makedirs('./cache', exist_ok=True)
os.makedirs('./experiments_results/summary', exist_ok=True)
DATA = ['sat', 'u2', 'u4', 'google', 'bats']


def load_pmi(file_):
    def _format(_list):
        try:
            return '{}-{}'.format(_list[0], _list[1]), float(_list[2])
        except Exception:
            return None
    with open(file_, 'r') as f:
        return dict(list(filter(None, [_format(i_.split('\t')) for i_ in f.read().split('\n') if len(i_) > 0])))


output = {}
for i in DATA:
    pmi_1 = load_pmi('./data/{}/pmi/pmi.1.csv'.format(i))
    pmi_2 = load_pmi('./data/{}/pmi/pmi.0.5.csv'.format(i))
    output[i] = {}
    for pmi, pmi_prefix in zip([pmi_1, pmi_2], ['pmi.1', 'pmi.0.5']):
        val, test = alm.data_analogy.get_dataset_raw(i)
        output[i][pmi_prefix] = {}
        for data, prefix in zip([val, test], ['valid', 'test']):
            line_oov = []
            line_accuracy = []
            for d in data:
                r_guess = randint(0, len(d['choice']) - 1)
                score = [pmi['-'.join(c)] if '-'.join(c) in pmi.keys() else -100 for c in d['choice']]
                pred = score.index(max(score))
                if pred == -100:
                    pred = r_guess
                line_oov += ['-'.join(c) for c in d['choice'] if '-'.join(c) not in pmi.keys()]
                line_accuracy.append(pred == d['answer'])
                if prefix == 'test':
                    d['prediction'] = pred
            if prefix == 'test':
                pd.DataFrame(data).to_csv('experiments_results/summary/statistics.{}.prediction.{}.{}.csv'.format(prefix, pmi_prefix, i))
            output[i][pmi_prefix][prefix] = {'accuracy': sum(line_accuracy)/len(line_accuracy), 'oov': len(line_oov)}

for prefix in ['valid', 'test']:
    df = pd.read_csv('./experiments_results/summary/statistics.{}.csv'.format(prefix), index_col=0)
    df_oov = pd.read_csv('./experiments_results/summary/statistics.{}.oov.csv'.format(prefix), index_col=0)
    for pmi_prefix in ['pmi.1', 'pmi.0.5']:
        list_a = []
        list_o = []
        for i in DATA:
            out = output[i][pmi_prefix][prefix]
            list_a.append(out['accuracy'])
            list_o.append(out['oov'])

        df[pmi_prefix] = list_a
        df_oov[pmi_prefix] = list_o
    df.to_csv('experiments_results/summary/statistics.{}.csv'.format(prefix))
    df_oov.to_csv('experiments_results/summary/statistics.{}.oov.csv'.format(prefix))
