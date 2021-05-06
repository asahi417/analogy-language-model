""" Solve analogy task by word embedding model """
import os
import logging
from itertools import chain

import alm
import pandas as pd
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
alm.util.fix_seed(1234)
os.makedirs('./cache', exist_ok=True)
os.makedirs('./experiments_results/summary', exist_ok=True)
DUMMY = -1000
if not os.path.exists('./cache/wiki-news-300d-1M.vec'):
    logging.info('downloading fasttext model')
    alm.util.open_compressed_file(
        url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
        cache_dir='./cache')
model = KeyedVectors.load_word2vec_format('./cache/wiki-news-300d-1M.vec')


def get_embedding(word_list):
    return dict(list(filter(lambda x: x[1] is not None, [(_i, model[_i] if _i in model.vocab else None) for _i in word_list])))


def cos_similarity(a_, b_):
    if a_ is None or b_ is None:
        return DUMMY
    inner = (a_ * b_).sum()
    norm_a = (a_ * a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    if norm_a == 0 or norm_b == 0:
        return DUMMY
    return inner / (norm_b * norm_a)


def get_prediction(stem, choice, embedding_dict):
    def diff(x, y):
        if x is None or y is None:
            return None
        return x - y

    stem_e = diff(embedding_dict[stem[0]] if stem[0] in embedding_dict else None,
                  embedding_dict[stem[1]] if stem[1] in embedding_dict else None)
    if stem_e is None:
        return None
    choice_e = [diff(embedding_dict[a] if a in embedding_dict else None, embedding_dict[b] if b in embedding_dict else None) for a, b in choice]
    score = [cos_similarity(e, stem_e) for e in choice_e]
    pred = score.index(max(score))
    if score[pred] == DUMMY:
        return None
    return pred


if __name__ == '__main__':
    for i in ['sat', 'u2', 'u4', 'google', 'bats']:
        _, data = alm.data_analogy.get_dataset_raw(i)
        answer = {n: o['answer'] for n, o in enumerate(data)}
        pmi_pred = {n: o['pmi_pred'] for n, o in enumerate(data)}
        dict_ = get_embedding(list(set(list(chain(*[list(chain(*[o['stem']] + o['choice'])) for o in data])))))
        prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(data)}
        prefix = 'experiments_results/summary/prediction_file/experiment'

        for n, d in enumerate(data):
            d['prediction'] = prediction[n] if prediction[n] is not None else pmi_pred[n]
        pd.DataFrame(data).to_csv('{}.word_embedding.test.prediction.{}.fasttext.csv'.format(prefix, i))

        for n, d in enumerate(data):
            d['prediction'] = pmi_pred[n]
        pd.DataFrame(data).to_csv('{}.pmi.test.prediction.{}.csv'.format(prefix, i))
