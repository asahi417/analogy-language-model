""" compile txt based dataset into jsonlines (duplication is also dropped)"""
import re
import json
import string
from copy import deepcopy
from glob import glob
from itertools import chain

ALPHABET_LIST = list(string.ascii_lowercase)
EXPORT_FILES = {
    './data/u4_raw.jsonl': {
        "high-advanced": "./data/original/u4/high-advanced*.txt",
        "high-beginning": "./data/original/u4/high-beginning*.txt",
        "high-intermediate": "./data/original/u4/high-intermediate*.txt",
        "low-advanced": "./data/original/u4/low-advanced*.txt",
        "low-intermediate": "./data/original/u4/low-intermediate*.txt",
    },
    './data/u2_raw.jsonl': {
        "grade4": "./data/original/u2/grade4*.txt",
        "grade5": "./data/original/u2/grade5*.txt",
        "grade6": "./data/original/u2/grade6*.txt",
        "grade7": "./data/original/u2/grade7*.txt",
        "grade8": "./data/original/u2/grade8*.txt",
        "grade9": "./data/original/u2/grade9*.txt",
        "grade10": "./data/original/u2/grade10*.txt",
        "grade11": "./data/original/u2/grade11*.txt",
        "grade12": "./data/original/u2/grade12*.txt"
    }
}

EXPORT_FILES_SAT = {'./data/sat_package_v3.jsonl': './data/original/SAT-package-V3.txt'}


def process_single_entry_sat(single_entry: str):
    lines = [re.sub(r'\s\Z', '', t) for t in single_entry.split('\n')]
    lines = list(filter(lambda x: len(x) > 0, lines))
    prefix = lines.pop(0)
    target = (lines[0].split(' ')[0].lower(), lines[0].split(' ')[1].lower())
    answer_id = ALPHABET_LIST.index(lines[-1])
    choice_list = [(__i.split(' ')[0], __i.split(' ')[1]) for __i in lines[1:-1]]
    return {"stem": target, "answer": answer_id, "choice": choice_list, "prefix": prefix}


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


def main(export, files, all_data=None, dup: int = 0):
    all_data = [] if all_data is None else all_data
    jsonl_file = format_data(files)
    tmp = []
    for __k in jsonl_file:
        i = deepcopy(__k)
        i.pop('prefix')
        if i in all_data:
            dup += 1
        else:
            all_data.append(i)
            tmp.append(__k)

    print('dataset: {}, length: {}, duplicate: {}'.format(export, len(tmp), dup))
    tmp = check_validity(tmp)
    with open(export, 'w') as writer:
        writer.write('\n'.join([json.dumps(d) for d in tmp]))
    return all_data, dup


if __name__ == '__main__':
    all_data_, dup_ = main('./data/u2_raw.jsonl', EXPORT_FILES['./data/u2_raw.jsonl'])
    main('./data/u4_raw.jsonl', EXPORT_FILES['./data/u4_raw.jsonl'], all_data_, dup_)

    for k, v in EXPORT_FILES_SAT.items():
        processed = [json.dumps(process_single_entry_sat(i)) for i in re.split(r'\n[\s]*\n', open(v, "r").read())[1:]]
        with open(k, 'w') as writer_:
            writer_.write('\n'.join(processed))
