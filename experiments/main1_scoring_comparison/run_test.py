""" Run on test set """
import logging
import json
from itertools import product
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 128), ('bert-large-cased', 32, 1024)]

#######################################################################
# get test accuracy on each combination of model and scoring function #
#######################################################################
methods = [
    'pmi_feldman', 'embedding_similarity', 'ppl', 'ppl_based_pmi', 'ppl_head_masked', 'ppl_tail_masked',
    'ppl_add_masked', 'ppl_marginal_bias', 'ppl_hypothesis_bias'
]
export_prefix = 'main1.scoring_comparison'
df = alm.get_report(export_prefix=export_prefix)

for i, m, s in product(data, models, methods):
    _model, _len, _batch = m
    tmp_df = df[df.data == i]
    tmp_df = tmp_df[tmp_df.model == _model]
    tmp_df = tmp_df[tmp_df.scoring_method == s]
    val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    logging.info("RUN TEST:\n - data: {} \n - lm: {} \n - score: {} \n - validation accuracy: {} ".format(
        i, _model, s, val_accuracy))
    best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
    logging.info("find {} configs with same accuracy".format(len(best_configs)))
    for n, tmp_df in best_configs.iterrows():
        config = json.loads(tmp_df.to_json())
        config.pop('accuracy')
        config.pop('max_length')
        scorer = alm.RelationScorer(model=config.pop('model'), max_length=_len)
        scorer.analogy_test(
            test=True, export_prefix=export_prefix, batch_size=_batch, val_accuracy=val_accuracy, **config
        )
        scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)

###################################################################################
# get test accuracy on each default configuration of model and (ppl, and ppl_pmi) #
###################################################################################
shared = {
    'export_prefix': 'main1_scoring_comparison.default',
    'template_type': 'is-to-as',
    'scoring_method': 'ppl_pmi',
}

for i, m in product(data, models):
    _model, _len, _batch = m
    scorer = alm.RelationScorer(model=_model, max_length=_len)
    val_accuracy = scorer.analogy_test(batch_size=_batch, data=i, test=False, **shared)
    assert len(val_accuracy) == 0
    scorer.analogy_test(batch_size=_batch, data=i, val_accuracy=val_accuracy, test=True, **shared)
    scorer.release_cache()

alm.export_report(export_prefix='main1_scoring_comparison.default', test=True)
