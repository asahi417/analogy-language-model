import alm

export_dir = './experiments/pmi_grid/results'
templates = [
    ['what-is-to'],  # u2/u4 first, sat second
    ['rel-same'],  # u2 second
    ['as-what-same']  # u4 second, sat first
]
pmi_lambdas = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]


def main(path_to_data):
    for template in templates:
        for pmi_lambda in pmi_lambdas:
            scorer = alm.RelationScorer(model='roberta-large', max_length=32)
            scorer.analogy_test(
                scoring_method='pmi',
                path_to_data=path_to_data,
                template_types=template,
                batch_size=512,
                export_dir=export_dir,
                permutation_negative=False,
                skip_scoring_prediction=True,
                pmi_lambda=pmi_lambda
            )
            scorer.release_cache()


if __name__ == '__main__':
    # tuning on each config
    main(path_to_data='./data/sat_package_v3.jsonl')
    main(path_to_data='./data/u2_raw.jsonl')
    main(path_to_data='./data/u4_raw.jsonl')

