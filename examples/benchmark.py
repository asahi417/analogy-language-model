""" Run scoring over all the baseline data """
import alm
import os
import json
from glob import glob
import pandas as pd

skip_negative = True

pmi_aggregations = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10', 'p_11']
all_templates = [['is-to-what'], ['is-to-as'], ['rel-same'], ['what-is-to'], ['she-to-as'], ['as-what-same']]
aggregation_positives = ['max', 'mean', 'min', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7']
data = ['./data/sat_package_v3.jsonl', './data/u2.jsonl', './data/u4.jsonl']
# lm = [('roberta-large', 32), ('gpt2-large', 32), ('gpt2-xl', 32), ('bert-large-cased', 64)]
lm = [('roberta-large', 32), ('gpt2-xl', 32)]
export_dir = './results'
shared_param = {'no_inference': True, 'overwrite_output': True, 'export_dir': export_dir}
# all_templates = [['as-what-same']]
# aggregation_positives = ['p_2']
# data = ['./data/sat_package_v3.jsonl']
# lm = [('roberta-large', 32)]


def main(path_to_data,
         model_type,
         max_length,
         scoring_method,
         pmi_aggregation=None,
         ppl_pmi_lambda=1.0):

    scorer = alm.RelationScorer(model=model_type, max_length=max_length)

    def sub(_template_types, _permutation_negative, _aggregation_positive):
        if _permutation_negative:
            for aggregation_negative in ['max', 'mean', 'min']:
                scorer.analogy_test(
                    scoring_method=scoring_method,
                    pmi_aggregation=pmi_aggregation,
                    path_to_data=path_to_data,
                    template_types=_template_types,
                    aggregation_positive=_aggregation_positive,
                    permutation_negative=_permutation_negative,
                    aggregation_negative=aggregation_negative,
                    ppl_pmi_lambda=ppl_pmi_lambda,
                    **shared_param
                )

        else:
            scorer.analogy_test(
                scoring_method=scoring_method,
                pmi_aggregation=pmi_aggregation,
                path_to_data=path_to_data,
                template_types=_template_types,
                aggregation_positive=_aggregation_positive,
                ppl_pmi_lambda=ppl_pmi_lambda,
                **shared_param
            )

    for template_types in all_templates:
        for aggregation_positive in aggregation_positives:
            sub(template_types, False, aggregation_positive)
            if not skip_negative:
                sub(template_types, True, aggregation_positive)


def summary():

    index = [
        'model', 'max_length', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive',
        'permutation_negative', 'aggregation_negative', 'pmi_aggregation', 'pmi_lambda', 'ppl_pmi_lambda']
    df = pd.DataFrame(index=index + ['accuracy'])

    for i in glob('results/outputs/*'):
        with open(os.path.join(i, 'accuracy.json'), 'r') as f:
            accuracy = json.load(f)
        with open(os.path.join(i, 'config.json'), 'r') as f:
            config = json.load(f)
        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                        [round(accuracy['accuracy'] * 100, 2)]

    df = df.T
    df = df.sort_values(by=index, ignore_index=True)
    df.to_csv('{}/summary.csv'.format(export_dir))


if __name__ == '__main__':

    # main('./data/sat_package_v3.jsonl', 'roberta-large', 32, scoring_method='ppl')
    # for i in range(-5, 5):
    #     main('./data/sat_package_v3.jsonl', 'roberta-large', 32, scoring_method='ppl_pmi', ppl_pmi_lambda=i*0.1)
    # for _i in [-2, -1.9, -1.8, -1.7, -1.6, -1.5]:
    #     main('./data/sat_package_v3.jsonl', 'roberta-large', 32, scoring_method='ppl_pmi', ppl_pmi_lambda=_i * 0.1)
    # summary()
    # exit()

    for _model, _max_length in lm:
        for _data in data:
            main(_data, _model, _max_length, scoring_method='ppl_pmi')
            main(_data, _model, _max_length, scoring_method='ppl')
            main(_data, _model, _max_length, scoring_method='embedding_similarity')
            if 'gpt' in _model:
                continue
            for m in pmi_aggregations:
                main(_data, _model, _max_length, scoring_method='pmi', pmi_aggregation=m)

    summary()
