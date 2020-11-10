import argparse
import os
import logging
import json

from tqdm import tqdm
from transformers_lm import TransformersLM
from prompting_relation import prompting_relation


LOGGER = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
LOGGER_DIR = './log'
VALID_DATA = {
    'sat': './data/sat_package_v3.jsonl',
    'u2': './data/sat_package_v3.jsonl',
    'u4': './data/sat_package_v3.jsonl'
}
if not os.path.exists(LOGGER_DIR):
    os.makedirs(LOGGER_DIR, exist_ok=True)


def main(dataset: str, method: str, lm_name: str, max_length: int):

    # add file handler
    file_handler = logging.FileHandler(
        os.path.join(LOGGER_DIR, 'experiment.{}.{}.{}.log'.format(lm_name, dataset, method)))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    LOGGER.addHandler(file_handler)

    # model setup
    LOGGER.info("ANALOGY PROBING EXPERIMENT")
    path_to_dataset = VALID_DATA[dataset]

    def format_entry(dictionary):
        prompts = [
            prompting_relation(
                subject_stem=dictionary['stem'][0],
                object_stem=dictionary['stem'][1],
                subject_analogy=c[0],
                object_analogy=c[1],
            ) for c in dictionary['choice']]

        return dictionary['answer'], prompts

    with open(path_to_dataset, 'r') as f:
        data = [
            format_entry(json.loads(i))
            for i in f.read().split('\n') if len(i) > 0]

    lm = TransformersLM(lm_name, max_length=max_length)
    accuracy = []
    for true, choice in tqdm(data):
        if method == 'pseudo-ppl':
            # use the lowest ppl as a prediction
            ppl = lm.get_pseudo_perplexity(choice)
            predict = ppl.index(min(ppl))
        else:
            raise ValueError('unknown method: {}'.format(method))
        # elif method == 'object-likelihood':
        #     lm.get_log_likelihood(choice, [])
        accuracy.append(predict == ppl)
        break
    print(accuracy)
    print(sum(accuracy)/len(accuracy))


def get_options():
    parser = argparse.ArgumentParser(description='run experiments on SAT dataset')
    parser.add_argument('-l', '--lm', help='language model', default='roberta-base', type=str)
    parser.add_argument('-d', '--dataset', help='dataset from {}'.format(VALID_DATA.keys()), default='sat', type=str)
    parser.add_argument('-b', '--batch-size', help='batch size', default=8, type=int)
    parser.add_argument('-m', '--method', help='method', default='pseudo-ppl', type=str)
    parser.add_argument('--max-length', help='max length', default=128, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    assert opt.dataset in VALID_DATA.keys()

    main(dataset=opt.dataset,
         method=opt.method,
         lm_name=opt.lm,
         max_length=opt.max_length)
