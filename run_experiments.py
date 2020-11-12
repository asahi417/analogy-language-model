""" run an analogy experiment """
import argparse
import os
import logging
import json
from itertools import chain

from transformers_lm import TransformersLM
from prompting_relation import get_prompt_dataset


LOGGER = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
VALID_DATA = {
    'sat': 'sat_package_v3.jsonl',
    'u2': 'u2.jsonl',
    'u4': 'u4.jsonl'
}


def get_random_score(dataset_path):
    """ Random baseline is 0.241 (u2.jsonl), 0.2 (sat_package_v3.jsonl), 0.235 (u4.jsonl) """
    with open(dataset_path, 'r') as f:
        choice_length = [1/len(json.loads(i) ['choice']) for i in f.read().split('\n') if len(i) > 0]
        score = sum(choice_length)/len(choice_length)
    return score


def main(dataset: str,
         method: str,
         lm_name: str,
         max_length: int,
         batch_size: int = 6,
         template_type: str = 'what-is',
         data_dir: str = './data',
         export_dir: str = './results',
         cache_dir: str = './cache'):
    """ run an analogy experiment

    :param dataset:
    :param method:
    :param lm_name:
    :param max_length:
    :param batch_size:
    :param template_type:
    """
    LOGGER.info("ANALOGY PROBING EXPERIMENT")

    # data setup
    dataset_path = os.path.join(data_dir, VALID_DATA[dataset])
    list_answer, list_nested_sentence, list_stem, list_nested_choice \
        = get_prompt_dataset(dataset_path, template_type=template_type)

    # multiple answer per question -> flatten to run inferences -> convert scores into nested structure
    length = [len(i) for i in list_nested_sentence]
    partition = [[sum(length[:i]), sum(length[:i + 1])] for i in range(len(length))]
    list_sentence = list(chain(*list_nested_sentence))

    # run an inference over all flatten sentences (lower score should be better in any cases)
    lm = TransformersLM(lm_name, max_length=max_length, cache_dir=cache_dir)
    if method == 'pseudo-ppl':
        list_score = lm.get_pseudo_perplexity(list_sentence, batch_size=batch_size)
    elif 'object-nll' in method:
        if 'object-nll-stem' in method:
            if 'object-nll-stem-subj' == method:
                list_target = list(list(zip(*list_stem))[0])
            elif 'object-nll-stem-obj' == method:
                list_target = list(list(zip(*list_stem))[1])
            else:
                raise ValueError('unknown method: {}'.format(method))
            list_target = list(chain(*[[i] * len(c) for i, c in zip(list_target, list_nested_sentence)]))
        elif 'object-nll-choice' in method:
            list_choice = list(chain(*list_nested_choice))
            if 'object-nll-choice-subj' == method:
                list_target = list(list(zip(*list_choice))[0])
            elif 'object-nll-choice-obj' == method:
                list_target = list(list(zip(*list_choice))[1])
            else:
                raise ValueError('unknown method: {}'.format(method))
        else:
            raise ValueError('unknown method: {}'.format(method))
        assert len(list_target) == len(list_sentence)
        list_score, _ = lm.get_nll(list_sentence, target_tokens=list_target, batch_size=batch_size)
    else:
        raise ValueError('unknown method: {}'.format(method))

    # reconstruct question-wise nested structure
    list_nested_scores = [list_score[s:e] for s, e in partition]
    assert len(list_nested_scores) == len(list_nested_sentence) == len(list_answer)
    list_prediction = [s.index(min(s)) for s in list_nested_scores]
    accuracy = sum(t == p for t, p in zip(list_answer, list_prediction))/len(list_answer)

    # export result
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    prefix = '{}.{}.{}.{}.{}'.format(lm_name, max_length, dataset, method, template_type)
    with open(os.path.join(export_dir, '{}.prediction'.format(prefix)), 'w') as f:
        f.write('true,prediction\n')
        f.write('\n'.join(['{},{}'.format(t, p) for t, p in zip(list_answer, list_prediction)]))
    with open(os.path.join(export_dir, '{}.accuracy'.format(prefix)), 'w') as f:
        f.write('{}'.format(accuracy))
    LOGGER.info('accuracy: {}'.format(accuracy))


def get_options():
    parser = argparse.ArgumentParser(description='run experiments on SAT dataset')
    parser.add_argument('-l', '--lm', help='language model', default='roberta-base', type=str)
    parser.add_argument('-d', '--dataset', help='dataset from {}'.format(VALID_DATA.keys()), default='sat', type=str)
    parser.add_argument('-b', '--batch-size', help='batch size', default=8, type=int)
    parser.add_argument('-m', '--method', help='method', default='pseudo-ppl', type=str)
    parser.add_argument('-t', '--template-type', help='template type', default='what-is', type=str)
    parser.add_argument('--max-length', help='max length', default=128, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    assert opt.dataset in VALID_DATA.keys()

    main(dataset=opt.dataset,
         method=opt.method,
         lm_name=opt.lm,
         max_length=opt.max_length,
         batch_size=opt.batch_size,
         template_type=opt.template_type)

