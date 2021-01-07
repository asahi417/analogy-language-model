import alm
import os
import json
from glob import glob
import pandas as pd

export_dir = './experiments/ppl_pmi_grid/results'
aggregation_ppl_pmi_whole = ['max', 'mean', 'min', 'p_0', 'p_1']
lambdas = list(map(lambda x: x/10, range(10, 21)))
alphas = list(map(lambda x: x / 10, range(-10, 1)))


def main(path_to_data, template, aggregation_positive):

    for ppl_pmi_aggregation in aggregation_ppl_pmi_whole:
        # get accuracy
        scorer = alm.RelationScorer(model='roberta-large', max_length=32)
        for lam in lambdas:
            for alpha in alphas:
                scorer.analogy_test(
                    scoring_method='ppl_pmi',
                    path_to_data=path_to_data,
                    ppl_pmi_aggregation=ppl_pmi_aggregation,
                    template_types=[template],
                    aggregation_positive=aggregation_positive,
                    ppl_pmi_lambda=lam,
                    ppl_pmi_alpha=alpha,
                    no_inference=True,
                    export_dir=export_dir
                )


if __name__ == '__main__':
    # main(path_to_data='./data/sat_package_v3.jsonl',
    #      template='as-what-same', aggregation_positive='p_2')
    # main(path_to_data='./data/u2.jsonl',
    #      template='she-to-as', aggregation_positive='p_0')
    # main(path_to_data='./data/u4.jsonl',
    #      template='what-is-to', aggregation_positive='p_0')

    # export as a csv
    index = ['model', 'path_to_data', 'scoring_method', 'template_types', 'aggregation_positive', 'ppl_pmi_lambda',
             'ppl_pmi_alpha', 'ppl_pmi_aggregation']
    df = pd.DataFrame(index=index + ['accuracy'])

    for i in glob('{}/outputs/*'.format(export_dir)):

        with open(os.path.join(i, 'config.json'), 'r') as f:
            config = json.load(f)
            if config['ppl_pmi_lambda'] not in lambdas:
                continue
            if config['ppl_pmi_alpha'] not in alphas:
                continue

        with open(os.path.join(i, 'accuracy.json'), 'r') as f:
            accuracy = json.load(f)
        df[len(df.T)] = [','.join(config[i]) if type(config[i]) is list else config[i] for i in index] + \
                        [round(accuracy['accuracy'] * 100, 2)]

    df = df.T
    df = df.sort_values(by=index, ignore_index=True)
    df.to_csv('{}/summary.close.csv'.format(export_dir))

