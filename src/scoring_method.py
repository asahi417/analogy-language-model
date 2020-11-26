import logging
from typing import List
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from .lm import TransformersLM
from .prompting_relation import prompting_relation, TEMPLATES

AGGREGATION = {'mean': lambda x: sum(x)/len(x), 'max': lambda x: max(x), 'min': lambda x: min(x)}


class RelationScorer:

    def __init__(self,
                 scoring_method: str = 'ppl',
                 template_types: List = None,
                 permutation_positive: bool = False,
                 permutation_negative: bool = False,
                 aggregation_positive: str = None,
                 aggregation_negative: str = None,
                 model: str = 'roberta-base',
                 max_length: int = None,
                 cache_dir: str = './cache',
                 num_worker: int = 4):
        """

        :param template_types: a list of templates for prompting
        :param permutation_positive: if utilize positive permutation
        :param permutation_negative: if utilize negative permutation
        :param aggregation_positive: aggregation method for positive permutations (`mean`, `max`, `min`)
        :param aggregation_negative: aggregation method for negative permutations (`mean`, `max`, `min`)
        :param model: LM parameter
        :param max_length: LM parameter
        :param cache_dir: LM parameter
        :param num_worker: LM parameter
        """
        logging.info("*** setting up a scorer ***")

        # sanity check
        assert permutation_positive == aggregation_positive is not None
        assert permutation_negative == aggregation_negative is not None
        if aggregation_positive:
            assert aggregation_positive in ['mean', 'max', 'min']
            self.aggregation_positive = AGGREGATION[aggregation_positive]
        else:
            self.aggregation_positive = None
        if aggregation_negative:
            assert aggregation_negative in ['mean', 'max', 'min']
            self.aggregation_negative = AGGREGATION[aggregation_negative]
        else:
            self.aggregation_negative = None
        if template_types is None:
            self.template_types = list(TEMPLATES.keys())
        else:
            assert all(t in TEMPLATES.keys() for t in template_types)
            self.template_types = template_types
        self.scoring_method = scoring_method
        # language model setup
        self.lm = TransformersLM(model=model, max_length=max_length, cache_dir=cache_dir, num_worker=num_worker)

    def batch_predict(self,
                batch_stem_relation_pairs: List,
                batch_choice_relation_pairs: List,
                batch_size: int = 16):
        """

        :param batch_stem_relation_pairs: a batch of stem relation pairs
        :param batch_choice_relation_pairs: a batch of choice relation pairs
        :param batch_size
        :return:
        """
        assert len(batch_choice_relation_pairs) == len(batch_choice_relation_pairs)

        if self.scoring_method == 'ppl':
            list_score = lm.get_pseudo_perplexity(list_sentence, batch_size=batch_size)
        else:
            raise ValueError('unknown scoring method: {}'.format(self.scoring_method))





