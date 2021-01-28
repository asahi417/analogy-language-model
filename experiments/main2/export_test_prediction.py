""" Export prediction file of the best configuration in test accuracy """
import logging
import json
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm


config = {
    'sat': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "mean",
        "ppl_pmi_alpha": 0,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_2",
        "negative_permutation_weight": 0.6,
        "data": "sat",
        "template_type": "what-is-to"
    },
    'u2': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "mean",
        "ppl_pmi_alpha": 0.4,
        "positive_permutation_aggregation": "index_0",
        "negative_permutation_aggregation": "index_3",
        "negative_permutation_weight": 0.6,
        "data": "u2",
        "template_type": "she-to-as"
    },
    'u4': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "index_1",
        "ppl_pmi_alpha": 0.4,
        "positive_permutation_aggregation": "index_4",
        "negative_permutation_aggregation": "index_5",
        "negative_permutation_weight": 0.6,
        "data": "u4",
        "template_type": "rel-same"
    },
    'google': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "min",
        "negative_permutation_aggregation": "index_7",
        "negative_permutation_weight": 0.6,
        "data": "google",
        "template_type": "what-is-to"
    },
    'bats': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": 0.4,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_10",
        "negative_permutation_weight": 0.6,
        "data": "bats",
        "template_type": "what-is-to"
    }
}
export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix, test=True)

for k, v in config.items():
    tmp_df = df[df.data == k]
    for k_, v_ in v.items():
        tmp_df = tmp_df[tmp_df[k_] == v_]

    val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
    logging.info("RUN TEST:\n - data: {} \n - validation accuracy: {} ".format(k, val_accuracy))
    best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
    logging.info("find {} configs with same accuracy".format(len(best_configs)))
    config = json.loads(best_configs.iloc[0].to_json())
    logging.info("use the first one: {} accuracy".format(config.pop('accuracy')))
    scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
    scorer.analogy_test(test=True,
                        export_prediction=True,
                        no_inference=True,
                        export_prefix=export_prefix,
                        **config)
    scorer.release_cache()
