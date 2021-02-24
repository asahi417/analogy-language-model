import logging
import json
from itertools import product
import alm
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
methods = ['pmi_feldman', 'embedding_similarity', 'ppl', 'ppl_based_pmi', 'ppl_head_masked', 'ppl_tail_masked']
data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 128), ('bert-large-cased', 32, 1024)]


logging.info('################################################')
logging.info('# Run LM inference to get logit (on valid set) #')
logging.info('################################################')
no_inference = True
for _model, _max_length, _batch in models:
    for scoring_method in methods:
        if 'gpt' in _model and scoring_method in ['ppl', 'ppl_based_pmi']:
            continue
        scorer = alm.RelationScorer(model=_model, max_length=_max_length)
        for _data in data:
            for _temp in all_templates:
                scorer.analogy_test(
                    scoring_method=scoring_method,
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    no_inference=no_inference,
                    skip_scoring_prediction=True)
                scorer.release_cache()


logging.info('#######################################################')
logging.info('# Get prediction on each configuration (on valid set) #')
logging.info('#######################################################')
methods += ['ppl_add_masked', 'ppl_marginal_bias', 'ppl_hypothesis_bias']
positive_permutation_aggregation = [
    'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7'
]
pmi_feldman_aggregation = [
    'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7',
    'index_8', 'index_9', 'index_10', 'index_11'
]
ppl_based_pmi_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1']
export_prefix = 'experiment.scoring_comparison'
no_inference = True
for _model, _max_length, _batch in models:
    for scoring_method in methods:
        if scoring_method in ['pmi', 'ppl_tail_masked', 'ppl_head_masked', 'ppl_add_masked', 'ppl_hypothesis_bias'] \
                and 'gpt' in _model:
            continue
        scorer = alm.RelationScorer(model=_model, max_length=_max_length)
        for _data in data:
            for _temp in all_templates:
                shared = dict(
                    scoring_method=scoring_method,
                    data=_data,
                    template_type=_temp,
                    batch_size=_batch,
                    export_prefix=export_prefix,
                    no_inference=no_inference,
                    positive_permutation_aggregation=positive_permutation_aggregation,
                    ppl_based_pmi_aggregation=ppl_based_pmi_aggregation
                )
                if scoring_method == 'pmi_feldman':
                    for i in pmi_feldman_aggregation:
                        scorer.analogy_test(pmi_feldman_aggregation=i, **shared)
                        scorer.release_cache()
                else:
                    scorer.analogy_test(**shared)
                scorer.release_cache()
alm.export_report(export_prefix=export_prefix)


logging.info('#######################################################################')
logging.info('# get test accuracy on each combination of model and scoring function #')
logging.info('#######################################################################')
no_inference = True
export_prefix = 'experiment.scoring_comparison'
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
            no_inference=no_inference,
            test=True,
            export_prefix=export_prefix,
            batch_size=_batch,
            val_accuracy=val_accuracy,
            **config
        )
        scorer.release_cache()
alm.export_report(export_prefix=export_prefix, test=True)


logging.info('############################################################')
logging.info('# get test accuracy on each default configuration of model #')
logging.info('############################################################')
no_inference = True
shared = {'template_type': 'is-to-as', 'scoring_method': 'ppl_based_pmi'}
export_prefix = 'experiment.scoring_comparison.default'
for i, m in product(data, models):
    _model, _len, _batch = m
    scorer = alm.RelationScorer(model=_model, max_length=_len)
    val_accuracy = scorer.analogy_test(batch_size=_batch, data=i, test=False, **shared)
    assert len(val_accuracy) == 0
    scorer.analogy_test(
        no_inference=no_inference,
        batch_size=_batch,
        data=i,
        val_accuracy=val_accuracy,
        export_prefix=export_prefix,
        test=True,
        **shared)
    scorer.release_cache()
alm.export_report(export_prefix=export_prefix, test=True)
