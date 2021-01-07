import alm

export_dir = './experiments/pmi_grid/results'


def main(path_to_data, template: str):
    for i in [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]:
        scorer = alm.RelationScorer(model='roberta-large', max_length=32)
        scorer.analogy_test(
            scoring_method='pmi',
            path_to_data=path_to_data,
            template_types=[template],
            batch_size=512,
            export_dir=export_dir,
            permutation_negative=False,
            skip_scoring_prediction=True,
            pmi_lambda=i
        )
        scorer.release_cache()


if __name__ == '__main__':
    main(path_to_data='./data/sat_package_v3.jsonl', template='rel-same')
    main(path_to_data='./data/u2.jsonl', template='rel-same')
    main(path_to_data='./data/u4.jsonl', template='what-is-to')
