""" Solve analogy task by word embedding model """
import os
import logging
from itertools import chain
from random import randint

import alm
import pandas as pd
from gensim.models import fasttext
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
alm.util.fix_seed(1234)

BIN_W2V = './cache/GoogleNews-vectors-negative300.bin'
BIN_FASTTEXT = './cache/crawl-300d-2M-subword.bin'
BIN_GLOVE = './cache/glove.840B.300d.txt'
BIN_GLOVE_W2V = './cache/glove_converted.txt'
os.makedirs('./cache', exist_ok=True)
os.makedirs('./experiments_results/summary', exist_ok=True)
DATA = ['sat', 'u2', 'u4', 'google', 'bats']
DUMMY = -1000
if not os.path.exists(BIN_W2V):
    logging.info('downloading word2vec model')
    alm.util.open_compressed_file(
        url="https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download", cache_dir='./cache', gdrive=True)
if not os.path.exists(BIN_FASTTEXT):
    logging.info('downloading fasttext model')
    alm.util.open_compressed_file(
        url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip', cache_dir='./cache')
if not os.path.exists(BIN_GLOVE):
    logging.info('downloading Glove model')
    alm.util.open_compressed_file(
        url='http://nlp.stanford.edu/data/glove.840B.300d.zip', cache_dir='./cache')
if not os.path.exists(BIN_GLOVE_W2V):
    logging.info('converting Glove model')
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
        embeddings = [(_i, embedding(_i, model_ft)) for _i in word_list]
    elif model_type == 'glove':
        embeddings = [(_i, embedding(_i, model_glove)) for _i in word_list]
    else:
        embeddings = [(_i, embedding(_i, model_w2v)) for _i in word_list]
    embeddings = list(filter(lambda x: x[1] is not None, embeddings))
    return dict(embeddings)


def get_prediction(stem, choice, embedding_dict):
    def diff(x, y):
        if x is None or y is None:
            return None
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
            val, test = alm.data_analogy.get_dataset_raw(i)
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

            if prefix == 'test':
                for n, d in enumerate(test):
                    d['prediction'] = ft_prediction[n]
                pd.DataFrame(test).to_csv('experiments_results/summary/word_embedding.test.prediction.{}.csv'.format(i))

        pd.DataFrame(line_accuracy).to_csv('experiments_results/summary/word_embedding.{}.csv'.format(prefix))
        pd.DataFrame(line_oov).to_csv('experiments_results/summary/word_embedding.{}.oov.csv'.format(prefix))


