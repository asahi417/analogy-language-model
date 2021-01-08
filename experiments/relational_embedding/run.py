"""
1) Get relational embedding model
normalized https://drive.google.com/file/d/1-w39MIMUkYuy2wdVGwOcgKimUV1vPOxk/view?usp=sharing
original https://drive.google.com/file/d/1HVJnTjcaQ3aCLdwTZwiGLpMDyEylx-zS/view?usp=sharing
"""
import json
from typing import List
from gensim.models import KeyedVectors

path = './cache/relative_wikipedia_en_300d.bin'
# path = './cache/fasttext_wikipedia_en_300d.bin
# path = './cache/relative-init_wikipedia_en_300d.bin'
model = KeyedVectors.load_word2vec_format(path, binary=True)
datasets = ['./data/sat_package_v3.jsonl', './data/u2.jsonl', './data/u4.jsonl']


def get_rel_embedding(relation: List):
    try:
        return model['__'.join(relation)]
    except KeyError:
        return None


def get_rel_similarity(stem: List, choice: List):
    stem_emb = get_rel_embedding(stem)
    choice_emb = get_rel_embedding(choice)
    if stem_emb is None or choice_emb is None:
        return None
    inner_prod = (stem_emb * choice_emb).sum()
    norm = (stem_emb ** 2).sum() ** 0.5 * (choice_emb ** 2).sum() ** 0.5
    return inner_prod / norm


def process_single(stem: List, choice: List):
    similarities = list(map(lambda x: get_rel_similarity(stem, x), choice))
    if all(map(lambda x: x is None, similarities)):
    # if any(map(lambda x: x is None, similarities)):
        return None
    else:
        return similarities


def get_dataset(path_to_data: str):
    """ Get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    with open(path_to_data, 'r') as f:
        return list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))


if __name__ == '__main__':
    for _data in datasets:
        print("Processing {}".format(_data))
        out = get_dataset(_data)
        answers = list(map(lambda x: x['answer'], out))
        scores = list(map(lambda x: process_single(x['stem'], x['choice']), out))
        in_vocab_n = len(list(filter(None, scores)))
        print('{}/{}'.format(in_vocab_n, len(out)))



