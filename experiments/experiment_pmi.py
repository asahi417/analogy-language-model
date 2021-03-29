import os
import logging
import alm
from random import randint, seed
import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
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


def get_prediction(pmi_, d):
    score = [pmi_['-'.join(c)] if '-'.join(c) in pmi_.keys() else -100 for c in d['choice']]
    return score.index(max(score))


if __name__ == '__main__':
    for prefix in ['test', 'valid']:
        line_oov = []
        line_accuracy = []
        for i in DATA:
            pmi = load_pmi('./data/{}/pmi/pmi.1.csv'.format(i))
            val, test = alm.data_analogy.get_dataset_raw(i)
            data = test if prefix == 'test' else val
            oov = {'data': i}
            all_accuracy = {'data': i}
            answer = {n: o['answer'] for n, o in enumerate(data)}
            random_prediction = {n: randint(0, len(o['choice']) - 1) for n, o in enumerate(data)}
            all_accuracy['random'] = sum([answer[n] == random_prediction[n] for n in range(len(answer))]) / len(answer)

            prediction = {n: get_prediction(pmi, o) for n, o in enumerate(data)}

            oov['pmi'] = 0
            for k, v in random_prediction.items():
                if prediction[k] == -100:
                    prediction[k] = v
                    oov['pmi'] += 1

            all_accuracy['pmi'] = sum([answer[n] == prediction[n] for n in range(len(answer))]) / len(answer)
            line_oov.append(oov)
            line_accuracy.append(all_accuracy)

            if prefix == 'test':
                for n, d in enumerate(data):
                    d['prediction'] = prediction[n]
                pd.DataFrame(data).to_csv(
                    'experiments_results/summary/prediction_file/'
                    'experiment.pmi.{}.prediction.{}.csv'.format(prefix, i))

        pd.DataFrame(line_accuracy).to_csv(
            'experiments_results/summary/experiment.pmi.{}.csv'.format(prefix))
        pd.DataFrame(line_oov).to_csv(
            'experiments_results/summary/experiment.pmi.{}.oov.csv'.format(prefix))



