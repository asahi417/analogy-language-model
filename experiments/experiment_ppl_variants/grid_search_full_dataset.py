import alm
from pprint import pprint

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat']
models = [('roberta-large', 32, 512), ('bert-large-cased', 32, 1024)]
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
scoring_method = ['ppl_hypothesis_bias', 'ppl_marginal_bias', 'ppl_based_pmi']
export_prefix = 'experiment.ppl_variants'
no_inference = True
###########################
# get valid/test accuracy #
###########################

for _model, _max_length, _batch in models:
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in all_templates:
            for test in [False, True]:
                for score in scoring_method:
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

########################
# merge into one table #
########################
df_val = alm.get_report(export_prefix=export_prefix)
df_test = alm.get_report(export_prefix=export_prefix, test=True)
for d in data:
    df_val = df_val[df_val.data == d]
    df_test = df_test[df_test.data == d]

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
