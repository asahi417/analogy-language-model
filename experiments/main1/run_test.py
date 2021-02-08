""" Test hypothesis only compared with Perplexity method """
import logging
import json
from itertools import product
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

data = ['sat', 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('bert-large-cased', 32, 1024)]

#######################################################################
# get test accuracy on each combination of model and scoring function #
#######################################################################
methods = ['ppl_pmi', 'ppl_tail_masked', 'ppl_head_masked', 'embedding_similarity', 'ppl_add_masked', 'pmi']
export_prefix = 'main1'
df = alm.get_report(export_prefix=export_prefix)

for i, m, s in product(data, models, methods):
    _model, _len, _batch = m
    tmp_df_ = df[df.data == i]
    tmp_df_ = tmp_df_[tmp_df_.model == _model]
    tmp_df_ = tmp_df_[tmp_df_.scoring_method == s]
    if s == 'ppl_pmi':
        ppl_pmi_marginal_version = [True, False]
    else:
        ppl_pmi_marginal_version = [False]
    for v in ppl_pmi_marginal_version:
        tmp_df = tmp_df_[tmp_df_.ppl_pmi_marginal_version == v]
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
                test=True, export_prefix=export_prefix, batch_size=_batch, val_accuracy=val_accuracy, **config)
            scorer.release_cache()

alm.export_report(export_prefix=export_prefix, test=True)

###################################################################################
# get test accuracy on each default configuration of model and (ppl, and ppl_pmi) #
###################################################################################
shared = {
    'export_prefix': 'main1.default',
    'template_type': 'is-to-as',
    'scoring_method': 'ppl_pmi',
    'positive_permutation_aggregation': 'index_0',
    'ppl_pmi_aggregation': 'mean',
    'ppl_pmi_alpha': 1.0
}

for i, m in product(data, models):
    for v in [True, False]:
        _model, _len, _batch = m
        scorer = alm.RelationScorer(model=_model, max_length=_len)
        val_accuracy = scorer.analogy_test(batch_size=_batch, data=i, ppl_pmi_marginal_version=v, test=False, **shared)
        print(val_accuracy)
        assert len(val_accuracy) == 0
        scorer.analogy_test(
            batch_size=_batch, data=i, ppl_pmi_marginal_version=v, val_accuracy=val_accuracy, test=True, **shared)
        scorer.release_cache()

alm.export_report(export_prefix='main1.default', test=True)
