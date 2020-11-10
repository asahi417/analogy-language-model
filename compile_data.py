""" compile txt based dataset into jsonlines """
import re
import json
import string

from glob import glob
from itertools import chain

ALPHABET_LIST = list(string.ascii_lowercase)
EXPORT_FILES = {
    './data/u2.jsonl': {
        "high-advanced": "./data/original/u4/high-advanced*.txt",
        "high-beginning": "./data/original/u4/high-beginning*.txt",
        "high-intermediate": "./data/original/u4/high-intermediate*.txt",
        "low-advanced": "./data/original/u4/low-advanced*.txt",
        "low-intermediate": "./data/original/u4/low-intermediate*.txt",
    },
    './data/u4.jsonl': {
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
    choice_list = [(i.split(' ')[0], i.split(' ')[1]) for i in lines[1:-1]]
    return {"stem": target, "answer": answer_id, "choice": choice_list, "prefix": prefix}


def process_single_entry(single_entry: str, level: str):
    lines = [re.sub(r'\s\Z', '', t) for t in single_entry.split('\n')]
    lines = list(filter(lambda x: len(x) > 0, lines))
    target = (lines[0].split(' ')[1].lower(), lines[0].split(' ')[-1].lower())
    answer_id = ALPHABET_LIST.index(lines[-1])
    choice_list = [(i.split(' ')[1], i.split(' ')[-1]) for i in lines[1:-1]]
    return {"stem": target, "answer": answer_id, "choice": choice_list, "prefix": level}


def format_data(txt_dir):
    all_data = []
    for _k, _v in txt_dir.items():
        all_data += list(chain(*[
            [process_single_entry(s, _k) for s in re.split(r'\n[\s]*\n', open(_i, "r").read())]
            for _i in glob(_v)]))

    return all_data


if __name__ == '__main__':
    for k, v in EXPORT_FILES.items():
        jsonl_file = format_data(v)
        with open(k, 'w') as writer:
            writer.write('\n'.join([json.dumps(d) for d in jsonl_file]))

    for k, v in EXPORT_FILES_SAT.items():
        processed = [json.dumps(process_single_entry_sat(i)) for i in re.split(r'\n[\s]*\n', open(v, "r").read())[1:]]
        with open(k, 'w') as writer:
            writer.write('\n'.join(processed))
