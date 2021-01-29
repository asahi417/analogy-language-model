import os
import alm
from itertools import chain
from random import randint, seed

import pandas as pd
from gensim.models import fasttext
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

seed(1234)
BIN_W2V = './cache/GoogleNews-vectors-negative300.bin'
BIN_FASTTEXT = './cache/crawl-300d-2M-subword.bin'
BIN_GLOVE = './cache/glove.840B.300d.txt'
BIN_GLOVE_W2V = './cache/glove_converted.txt'
os.makedirs('./cache', exist_ok=True)
os.makedirs('./experiments_results/summary', exist_ok=True)
DATA = ['sat', 'u2', 'u4', 'google', 'bats']
DUMMY = -1000
if not os.path.exists(BIN_W2V):
    raise ValueError('download embedding from "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit",'
                     'unzip, and put it as {}'.format(BIN_W2V))
if not os.path.exists(BIN_FASTTEXT):
    raise ValueError('download embedding from "https://fasttext.cc/docs/en/english-vectors.html",'
                     'unzip, and put it as {}'.format(BIN_FASTTEXT))
if not os.path.exists(BIN_GLOVE):
    raise ValueError('download embedding from "http://nlp.stanford.edu/data/glove.840B.300d.zip"'
                     'unzip, and put it as {}'.format(BIN_FASTTEXT))

if not os.path.exists(BIN_GLOVE_W2V):
    glove2word2vec(glove_input_file="./cache/glove.840B.300d.txt", word2vec_output_file="./cache/glove_converted.txt")
model_glove = KeyedVectors.load_word2vec_format(BIN_GLOVE_W2V)
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
    inner = (a_ * b_).sum()
    norm_a = (a_ * a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    return inner / (norm_b * norm_a)


def get_embedding(word_list, model_type=None):
    if model_type == 'fasttext':
        embeddings = [(_i, embedding(_i, model_w2v)) for _i in word_list]
    elif model_type == 'glove':
        embeddings = [(_i, embedding(_i, model_glove)) for _i in word_list]
    else:
        embeddings = [(_i, embedding(_i, model_ft)) for _i in word_list]
    embeddings = list(filter(lambda x: x[1] is not None, embeddings))
    return dict(embeddings)


def get_prediction(stem, choice, embedding_dict):
    def diff(x, y):
        if x is None or y is None:
            return None
        # return np.abs(x-y)
        return x - y

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
    for prefix in ['test', 'valid']:
        line_oov = []
        line_accuracy = []
        for i in DATA:
            oov = {'data': i}
            all_accuracy = {'data': i}
            val, test = alm.data.get_dataset_raw(i)
            if prefix == 'valid':
                test = val
            answer = {n: o['answer'] for n, o in enumerate(test)}
            random_prediction = {n: randint(0, len(o['choice']) - 1) for n, o in enumerate(test)}
            all_accuracy['random'] = sum([answer[n] == random_prediction[n] for n in range(len(answer))]) / len(answer)

            vocab = list(set(list(chain(*[list(chain(*[o['stem']] + o['choice'])) for o in test]))))

            dict_ = get_embedding(vocab)
            w2v_prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(test)}
            dict_ = get_embedding(vocab, model_type='fasttext')
            ft_prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(test)}
            dict_ = get_embedding(vocab, model_type='glove')
            glove_prediction = {n: get_prediction(o['stem'], o['choice'], dict_) for n, o in enumerate(test)}
            oov['w2v'] = 0
            oov['fasttext'] = 0
            oov['glove'] = 0
            for k, v in random_prediction.items():
                if w2v_prediction[k] is None:
                    w2v_prediction[k] = v
                    oov['fasttext'] += 1
                if ft_prediction[k] is None:
                    ft_prediction[k] = v
                    oov['w2v'] += 1
                if glove_prediction[k] is None:
                    glove_prediction[k] = v
                    oov['glove'] += 1

            all_accuracy['w2v'] = sum([answer[n] == w2v_prediction[n] for n in range(len(answer))]) / len(answer)
            all_accuracy['fasttext'] = sum([answer[n] == ft_prediction[n] for n in range(len(answer))]) / len(answer)
            all_accuracy['glove'] = sum([answer[n] == glove_prediction[n] for n in range(len(answer))]) / len(answer)
            line_oov.append(oov)
            line_accuracy.append(all_accuracy)
            print(all_accuracy)
            print(oov)

            if prefix == 'test':
                for n, d in enumerate(test):
                    d['prediction'] = ft_prediction[n]
                pd.DataFrame(test).to_csv('experiments_results/summary/statistics.test.prediction.{}.csv'.format(i))

        print(pd.DataFrame(line_accuracy))
        print(pd.DataFrame(line_oov))
        pd.DataFrame(line_accuracy).to_csv('experiments_results/summary/statistics.{}.csv'.format(prefix))
        pd.DataFrame(line_oov).to_csv('experiments_results/summary/statistics.{}.oov.csv'.format(prefix))


