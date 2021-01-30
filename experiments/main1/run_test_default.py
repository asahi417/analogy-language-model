""" Test default configuration """
import logging
import json
from itertools import product
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm


data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('bert-large-cased', 32, 1024), ('gpt2-xl', 32, 128)]

export_prefix = 'main1.default'

for i, m in product(data, models):
    _model, _len, _batch = m
    scorer = alm.RelationScorer(model=_model, max_length=_len)
    scorer.analogy_test(test=True,
                        template_type='is-to-as',
                        export_prefix=export_prefix,
                        batch_size=_batch,
                        data=i,
                        scoring_method='ppl',
                        positive_permutation_aggregation='index_0')
    scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)
