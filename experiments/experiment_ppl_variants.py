import logging
import json
from pprint import pprint
import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat']  #, 'u2', 'u4', 'google', 'bats']
models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]
scoring_method = ['ppl_hypothesis_bias', 'ppl_marginal_bias', 'ppl_based_pmi']

logging.info('###############################################################')
logging.info('# Run LM inference to get logit (both of valid and test sets) #')
logging.info('###############################################################')
no_inference = True
for _model, _max_length, _batch in models:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            for test in [True, False]:
                for score in scoring_method:
                    if 'gpt' in _model and score == 'ppl_hypothesis_bias':
                        continue
                    scorer.analogy_test(
                            scoring_method=score,
                            data=_data,
                            template_type=_temp,
                            batch_size=_batch,
                            no_inference=no_inference,
                            negative_permutation=True,
                            skip_scoring_prediction=True,
                            test=test
                        )
                    scorer.release_cache()

logging.info('######################################################################')
logging.info('# Get prediction on each configuration (both of valid and test sets) #')
logging.info('######################################################################')
positive_permutation_aggregation = [
    'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7'
]
negative_permutation_aggregation = [
    'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7',
    'index_8', 'index_9', 'index_10', 'index_11'
]
negative_permutation_weight = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
weight_head = [-0.4, -0.2, 0, 0.2, 0.4]
weight_tail = [-0.4, -0.2, 0, 0.2, 0.4]
ppl_based_pmi_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1']
ppl_based_pmi_alpha = [-0.4, -0.2, 0, 0.2, 0.4]
export_prefix = 'experiment.ppl_variants'
no_inference = True

for _model, _max_length, _batch in models:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            for test in [False, True]:
                for score in scoring_method:
                    if 'gpt' in _model and score == 'ppl_hypothesis_bias':
                        continue
                    scorer.analogy_test(
                        no_inference=no_inference,
                        scoring_method=score,
                        data=_data,
                        template_type=_temp,
                        batch_size=_batch,
                        export_prefix=export_prefix,
                        ppl_hyp_weight_head=weight_head,
                        ppl_hyp_weight_tail=weight_tail,
                        ppl_mar_weight_head=weight_head,
                        ppl_mar_weight_tail=weight_tail,
                        ppl_based_pmi_aggregation=ppl_based_pmi_aggregation,
                        ppl_based_pmi_alpha=ppl_based_pmi_alpha,
                        negative_permutation=True,
                        positive_permutation_aggregation=positive_permutation_aggregation,
                        negative_permutation_aggregation=negative_permutation_aggregation,
                        negative_permutation_weight=negative_permutation_weight,
                        test=test)
                    scorer.release_cache()

alm.export_report(export_prefix=export_prefix)
alm.export_report(export_prefix=export_prefix, test=True)

logging.info('merge into one table')
df_val = alm.get_report(export_prefix=export_prefix)
df_test = alm.get_report(export_prefix=export_prefix, test=True)
df_val = df_val.sort_values(by=list(df_val.columns))
df_test = df_test.sort_values(by=list(df_val.columns))

accuracy_val = df_val.pop('accuracy').to_numpy()
accuracy_test = df_test.pop('accuracy').to_numpy()
assert df_val.shape == df_test.shape

df_test['accuracy_validation'] = accuracy_val
df_test['accuracy_test'] = accuracy_test

df_test['accuracy'] = (accuracy_val * 37 + accuracy_test * 337)/(37 + 337)
df_test = df_test.sort_values(by=['accuracy'], ascending=False)
df_test.to_csv('./experiments_results/summary/{}.full.csv'.format(export_prefix))
pprint(df_test['accuracy'].head(10))


logging.info('###############################################')
logging.info('# Export predictions for qualitative analysis #')
logging.info('###############################################')
# get prediction of what achieves the best validation accuracy
score = 'ppl_based_pmi'
data = 'sat'
for _model, _max_length, _batch in models:
    tmp_df = df_test[df_test.data == data]
    tmp_df = tmp_df[tmp_df.model == _model]
    tmp_df = tmp_df[tmp_df.scoring_method == score]
    accuracy = tmp_df.sort_values(by='accuracy_validation', ascending=False).head(1)['accuracy'].values[0]
    best_configs = tmp_df[tmp_df['accuracy_validation'] == accuracy]
    config = json.loads(best_configs.iloc[0].to_json())
    logging.info("use the first one ({})".format(data))
    logging.info("\t * accuracy (valid): {}".format(config.pop('accuracy_validation')))
    logging.info("\t * accuracy (test) : {}".format(config.pop('accuracy_test')))
    logging.info("\t * accuracy (full) : {}".format(config.pop('accuracy')))
    scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
    scorer.analogy_test(test=True, export_prediction=True, no_inference=True, export_prefix=export_prefix, **config)
    scorer.release_cache()

