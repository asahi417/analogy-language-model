""" compile txt based dataset into jsonlines (duplication is also dropped)"""
import re
import json
import string
import os
from copy import deepcopy
from random import shuffle
ALPHABET_LIST = list(string.ascii_lowercase)


def process_single_entry_sat(single_entry: str):
    lines = [re.sub(r'\s\Z', '', t) for t in single_entry.split('\n')]
    lines = list(filter(lambda x: len(x) > 0, lines))
    prefix = lines.pop(0)
    target = (lines[0].split(' ')[0].lower(), lines[0].split(' ')[1].lower())
    answer_id = ALPHABET_LIST.index(lines[-1])
    choice_list = [(__i.split(' ')[0], __i.split(' ')[1]) for __i in lines[1:-1]]
    return {"stem": target, "answer": answer_id, "choice": choice_list, "prefix": prefix}


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


path = './data/sat/original/SAT-package-V3.txt'
processed = [process_single_entry_sat(i) for i in re.split(r'\n[\s]*\n', open(path, "r").read())[1:]]
processed = check_validity(processed)

#########################
# validation/test split #
#########################
shuffle(processed)
ratio = 0.1
val_size = int(len(processed) * ratio)
val = processed[:val_size]
test = processed[val_size:]

os.makedirs('./data/sat', exist_ok=True)
with open('./data/sat/valid.jsonl', 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in val]))
with open('./data/sat/test.jsonl', 'w') as writer:
    writer.write('\n'.join([json.dumps(d) for d in test]))
