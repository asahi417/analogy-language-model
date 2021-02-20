import os
import logging
import requests
import zipfile
import json
from typing import Dict

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
url = 'https://dl.fbaipublicfiles.com/LAMA/data.zip'
relations_google = [
    {
        "relation": "place_of_birth",
        "template": "[X] was born in [Y] .",
        "template_negated": "[X] was not born in [Y] .",
    },
    {
        "relation": "date_of_birth",
        "template": "[X] (born [Y]).",
        "template_negated": "[X] (not born [Y]).",
    },
    {
        "relation": "place_of_death",
        "template": "[X] died in [Y] .",
        "template_negated": "[X] did not die in [Y] .",
    },
]
relations_concept_squad = [{"relation": "test", "template": None}]


def parse_template(template, subject_label, object_label):
    return [template.replace("[X]", subject_label).replace("[Y]", object_label)]


def get_lama_data(cache_dir: str = './data/lama', vocab: Dict = None):
    if not os.path.exists('{}/data'.format(cache_dir)):
        logging.info('downloading zip file from {}'.format(url))
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(url)
        with open('{}/{}'.format(cache_dir, filename), "wb") as f:
            r = requests.get(url)
            f.write(r.content)

        with zipfile.ZipFile('{}/{}'.format(cache_dir, filename), 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove('{}/{}'.format(cache_dir, filename))

    full_set = {}

    def load_jsonl(__file):
        with open(__file, 'r') as _f:
            return list(filter(None, map(lambda x: json.loads(x) if len(x) else None, _f.read().split('\n'))))

    def get_value(_dict, template: str = None):
        try:
            if vocab:  # make sure obj_label is in vocabulary
                assert vocab[_dict['obj_label']]
            if template:
                _dict['masked_sentences'] = parse_template(template, _dict['sub_label'], _dict['obj_label'])
            else:
                assert len(_dict['masked_sentences']) == 1 and type(_dict['masked_sentences']) is list
                _dict['masked_sentences'] = _dict['masked_sentences'][0]
            return {k: _dict[k] for k in ['obj_label', 'sub_label', 'masked_sentences']}
        except KeyError:
            return None

    logging.info('processing data')

    for i in ['ConceptNet', 'Google_RE', 'Squad', 'TREx']:
        if i == 'TREx':
            relation = load_jsonl('{}/data/relations.jsonl'.format(cache_dir))
        elif i == 'Google_RE':
            relation = relations_google
        else:
            relation = relations_concept_squad

        full_set[i] = {}
        for r in relation:
            if i == 'Google_RE':
                _file = '{}/data/{}/{}_test.jsonl'.format(cache_dir, i, r['relation'])
            else:
                _file = '{}/data/{}/{}.jsonl'.format(cache_dir, i, r['relation'])

            if not os.path.exists(_file):
                logging.warning('file not found {}'.format(_file))
            else:
                data = list(filter(None, map(lambda x: get_value(x, template=r['template']), load_jsonl(_file))))
                full_set[i][r['relation']] = data
                logging.info('\t * {}/{}: {}'.format(i, r['relation'], len(data)))
        logging.info('\t * {}: {}'.format(i, sum(len(i) for i in full_set[i].values())))
    return full_set


if __name__ == '__main__':
    import transformers
    # t = transformers.AutoTokenizer.from_pretrained('roberta-large')
    t = transformers.AutoTokenizer.from_pretrained('bert-large-cased')
    get_lama_data(vocab=t.vocab)
