import os
import alm
from itertools import chain
from random import randint, seed

import numpy as np
import pandas as pd

seed(1234)
BIN_W2V = './cache/GoogleNews-vectors-negative300.bin'
BIN_FASTTEXT = './cache/crawl-300d-2M-subword.bin'
os.makedirs('./cache', exist_ok=True)
DATA = ['sat', 'u2', 'u4', 'google', 'bats']
DUMMY = -1000
if not os.path.exists(BIN_W2V):
    raise ValueError('download embedding from "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit",'
                     'unzip, and put it as {}'.format(BIN_W2V))

if not os.path.exists(BIN_FASTTEXT):
    raise ValueError('download embedding from "https://fasttext.cc/docs/en/english-vectors.html",'
                     'unzip, and put it as {}'.format(BIN_FASTTEXT))

from gensim.models import fasttext
from gensim.models import KeyedVectors

model_w2v = KeyedVectors.load_word2vec_format(BIN_W2V, binary=True)
model_ft = fasttext.load_facebook_model(BIN_FASTTEXT)

def embedding(term, model):
    try:
        return model[term]
    except Exception:
        return None


def cos_similarity(a_, b_):
    if a_ is None or b_ is None:
        return DUMMY
    inner = (a_*b_).sum()
    norm_a = (a_*a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    return inner / (norm_b * norm_a)


def get_embedding(word_list, fasttext: bool):
    if fasttext:
        embeddings = [(_i, embedding(_i, model_w2v)) for _i in word_list]
    else:
        embeddings = [(_i, embedding(_i, model_ft)) for _i in word_list]
    embeddings = list(filter(lambda x: x[1] is not None, embeddings))
    return dict(embeddings)


def get_prediction(stem, choice, embedding_dict):
    def diff(x, y):
        if x is None or y is None:
            return None
        return np.abs(x-y)

    stem_e = diff(embedding(stem[0], embedding_dict), embedding(stem[1], embedding_dict))
    if stem_e is None:
        return None
    choice_e = [diff(embedding(a, embedding_dict), embedding(b, embedding_dict)) for a, b in choice]
    score = [cos_similarity(e, stem_e) for e in choice_e]
    pred = score.index(max(score))
    if score[pred] == DUMMY:
        return None
    return pred


if __name__ == '__main__':
    line_oov = []
    line_accuracy = []
    for i in DATA:
        oov = {'data': i}
        all_accuracy = {'data': i}
        _, test = alm.data.get_dataset_raw(i)
        answer = {n: o['answer'] for n, o in enumerate(test)}
        random_prediction = {n: randint(0, len(o['choice']) - 1) for n, o in enumerate(test)}
        all_accuracy['random'] = sum([answer[n] == random_prediction[n] for n in range(len(answer))]) / len(answer)

        vocab = list(set(list(chain(*[list(chain(*[o['stem']] + o['choice'])) for o in test]))))

        dict_ = get_embedding(vocab, fasttext=False)
        w2v_prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(test)}
        oov['w2v'] = len([i for i in w2v_prediction.values() if i is None])
        dict_ = get_embedding(vocab, fasttext=True)
        ft_prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(test)}
        oov['fasttext'] = len([i for i in ft_prediction.values() if i is None])
        for k, v in random_prediction.items():
            if w2v_prediction[k] is None:
                w2v_prediction[k] = v
            if ft_prediction[k] is None:
                ft_prediction[k] = v

        all_accuracy['w2v'] = sum([answer[n] == random_prediction[n] for n in range(len(answer))]) / len(answer)
        all_accuracy['fasttext'] = sum([answer[n] == random_prediction[n] for n in range(len(answer))]) / len(answer)
        line_oov.append(oov)
        line_accuracy.append(all_accuracy)
        print(all_accuracy)
        print(oov)
        input()

    print(pd.DataFrame(line_accuracy))
    print(pd.DataFrame(line_oov))
    pd.DataFrame(line_accuracy).to_csv('experiments_results/summary/statistics.test.csv')
    pd.DataFrame(line_oov).to_csv('experiments_results/summary/statistics.test.oov.csv')


