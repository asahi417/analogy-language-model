""" compile txt based dataset into jsonlines (duplication is also dropped)"""
import re
import json
import os
import string
from copy import deepcopy
from glob import glob
from itertools import chain
from random import shuffle

ALPHABET_LIST = list(string.ascii_lowercase)
EXPORT_FILES = {
    'u4': {
        "high-advanced": "./data/u4/original/high-advanced*.txt",
        "high-beginning": "./data/u4/original/high-beginning*.txt",
        "high-intermediate": "./data/u4/original/high-intermediate*.txt",
        "low-advanced": "./data/u4/original/low-advanced*.txt",
        "low-intermediate": "./data/u4/original/low-intermediate*.txt",
    },
    'u2': {
        "grade4": "./data/u2/original/grade4*.txt",
        "grade5": "./data/u2/original/grade5*.txt",
        "grade6": "./data/u2/original/grade6*.txt",
        "grade7": "./data/u2/original/grade7*.txt",
        "grade8": "./data/u2/original/grade8*.txt",
        "grade9": "./data/u2/original/grade9*.txt",
        "grade10": "./data/u2/original/grade10*.txt",
        "grade11": "./data/u2/original/grade11*.txt",
        "grade12": "./data/u2/original/grade12*.txt"
    }
}


def process_single_entry(single_entry: str, level: str):
    lines_org = [re.sub(r'\s+\Z', '', t).replace('\u0301', '').replace('\\', '') for t in single_entry.split('\n')]
    lines = list(filter(lambda x: len(x) > 0, lines_org))
    target = (lines[0].split(' ')[1].lower(), lines[0].split(' ')[-1].lower())
    if target[0] == target[1]:
        target = target[0].split(':')
    target = list(map(lambda x: x.replace(':', ''), target))
    answer_id = ALPHABET_LIST.index(lines[-1])
    choice_list = [(__i.split(' ')[1].replace(':', ''), __i.split(' ')[-1].replace(':', '')) for __i in lines[1:-1]]
    for a, b in choice_list:
        if len(a) == 0 or len(b) == 0:
            raise ValueError(lines_org)

    return {"stem": target, "answer": answer_id, "choice": choice_list, "prefix": level}


def format_data(txt_dir):
    _all_data = []

    for _k, _v in txt_dir.items():
        _all_data += list(chain(*[
            [process_single_entry(s, _k) for s in re.split(r'\n[\s]*\n', open(_i, "r").read())]
            for _i in glob(_v)]))

    return _all_data


def check_validity(json_data):
    tmp = deepcopy(json_data)
    for t, i in enumerate(json_data):
        a = i['answer']
        choice = []
        for n, (c_h, c_t) in enumerate(i['choice']):
            if len(c_h) == 0 or len(c_t) == 0:
                raise ValueError(i)
            if c_h in i['stem'] or c_t in i['stem']:
                print("found duplication: {} \n {}".format((c_h, c_t), i))
                if a == n:
                    raise ValueError('answer would be dropped')
                if a > n:
                    a = a - 1
            else:
                choice.append((c_h, c_t))
        tmp[t]['answer'] = a
        tmp[t]['choice'] = choice
        assert tmp[t]['choice'][tmp[t]['answer']] == i['choice'][i['answer']], str((
                tmp[t]['choice'][tmp[t]['answer']], i['choice'][i['answer']]
        ))

    return tmp


for k, v in EXPORT_FILES.items():
    jsonl_file = format_data(v)
    analogy_data = check_validity(jsonl_file)

    #########################
    # validation/test split #
    #########################
    all_relation = list(set([i['prefix'] for i in jsonl_file]))
    val = []
    test = []
    ratio = 0.1
    for r in all_relation:
        sub = list(filter(lambda x: x['prefix'] == r, analogy_data))
        val_size = int(len(sub) * ratio)
        print('- relation: {}, val size: {}'.format(r, val_size))
        shuffle(sub)
        val += sub[:val_size]
        test += sub[val_size:]
    print('total     : {}'.format(len(analogy_data)))
    print('validation: {}'.format(len(val)))
    print('test      : {}'.format(len(test)))
    print('Dataset built: {}'.format(len(analogy_data)))

    os.makedirs('./data/{}'.format(k), exist_ok=True)
    with open('./data/{}/valid.jsonl'.format(k), 'w') as writer:
        writer.write('\n'.join([json.dumps(d) for d in val]))
    with open('./data/{}/test.jsonl'.format(k), 'w') as writer:
        writer.write('\n'.join([json.dumps(d) for d in test]))

