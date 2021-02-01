""" Export prediction file of the best configuration in test accuracy """
import logging
import json
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import alm

export_prefix = 'main2'
df = alm.get_report(export_prefix=export_prefix, test=True)


def main(config):
    for k, v in config.items():
        tmp_df = df[df.data == k]
        for k_, v_ in v.items():
            tmp_df = tmp_df[tmp_df[k_] == v_]

        val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
        logging.info("RUN TEST:\n - data: {} \n - validation accuracy: {} ".format(k, val_accuracy))
        best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
        assert len(best_configs) == 1, str(best_configs)
        config = json.loads(best_configs.iloc[0].to_json())
        logging.info("use the first one: {} accuracy".format(config.pop('accuracy')))
        scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'))
        scorer.analogy_test(test=True,
                            export_prediction=True,
                            no_inference=True,
                            export_prefix=export_prefix,
                            **config)
        scorer.release_cache()


# these are the most neutral configurations among what achieve the best validation accuracy (with RoBERTa)
config_gpt = {
    'sat': {
        "model": "gpt2-xl",
        "ppl_pmi_aggregation": "mean",
        "ppl_pmi_alpha": 0.2,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "min",
        "negative_permutation_weight": 0.4,
        "data": "sat",
        "template_type": "as-what-same"
    },
    'u2': {
        "model": "gpt2-xl",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": -0.2,
        "positive_permutation_aggregation": "min",
        "negative_permutation_aggregation": "index_3",
        "negative_permutation_weight": 0.4,
        "data": "u2",
        "template_type": "rel-same"
    },
    'u4': {
        "model": "gpt2-xl",
        "ppl_pmi_aggregation": "max",
        "ppl_pmi_alpha": 0.0,
        "positive_permutation_aggregation": "index_0",
        "negative_permutation_aggregation": "min",
        "negative_permutation_weight": 0,
        "data": "u4",
        "template_type": "is-to-what"
    },
    'google': {
        "model": "gpt2-xl",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "index_0",
        "negative_permutation_aggregation": "index_5",
        "negative_permutation_weight": 0.4,
        "data": "google",
        "template_type": "what-is-to"
    },
    'bats': {
        "model": "gpt2-xl",
        "ppl_pmi_aggregation": "index_0",
        "ppl_pmi_alpha": -0.2,
        "positive_permutation_aggregation": "index_0",
        "negative_permutation_aggregation": "index_4",
        "negative_permutation_weight": 0.6,
        "data": "bats",
        "template_type": "what-is-to"
    }
}


config_roberta = {
    'sat': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "mean",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_10",
        "negative_permutation_weight": 0.6,
        "data": "sat",
        "template_type": "she-to-as"
    },
    'u2': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "mean",
        "ppl_pmi_alpha": 0.0,
        "positive_permutation_aggregation": "index_0",
        "negative_permutation_aggregation": "index_6",
        "negative_permutation_weight": 0.4,
        "data": "u2",
        "template_type": "what-is-to"
    },
    'u4': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": 0.0,
        "positive_permutation_aggregation": "index_6",
        "negative_permutation_aggregation": "index_6",
        "negative_permutation_weight": 0.2,
        "data": "u4",
        "template_type": "she-to-as"
    },
    'google': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_10",
        "negative_permutation_weight": 0.6,
        "data": "google",
        "template_type": "what-is-to"
    },
    'bats': {
        "model": "roberta-large",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_10",
        "negative_permutation_weight": 0.6,
        "data": "bats",
        "template_type": "what-is-to"
    }
}

config_bert = {
    'sat': {
        "model": "berta-large-cased",
        "ppl_pmi_aggregation": "min",
        "ppl_pmi_alpha": -0.2,
        "positive_permutation_aggregation": "index_4",
        "negative_permutation_aggregation": "index_11",
        "negative_permutation_weight": 0.8,
        "data": "sat",
        "template_type": "as-what-same"
    },
    'u2': {
        "model": "berta-large-cased",
        "ppl_pmi_aggregation": "max",
        "ppl_pmi_alpha": 0.4,
        "positive_permutation_aggregation": "index_4",
        "negative_permutation_aggregation": "index_0",
        "negative_permutation_weight": 1,
        "data": "u2",
        "template_type": "rel-same"
    },
    'u4': {
        "model": "berta-large-cased",
        "ppl_pmi_aggregation": "max",
        "ppl_pmi_alpha": 0.2,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_0",
        "negative_permutation_weight": 0.6,
        "data": "u4",
        "template_type": "is-to-as"
    },
    'google': {
        "model": "berta-large-cased",
        "ppl_pmi_aggregation": "index_0",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "mean",
        "negative_permutation_aggregation": "index_10",
        "negative_permutation_weight": 0.4,
        "data": "google",
        "template_type": "what-is-to"
    },
    'bats': {
        "model": "berta-large-cased",
        "ppl_pmi_aggregation": "index_0",
        "ppl_pmi_alpha": -0.4,
        "positive_permutation_aggregation": "index_6",
        "negative_permutation_aggregation": "index_4",
        "negative_permutation_weight": 0.2,
        "data": "bats",
        "template_type": "as-what-same"
    }
}

main(config_gpt)
main(config_bert)
main(config_roberta)
