import alm

data = [
    # ('./data/u4.jsonl', 'what-is-to'),
    # ('./data/sat_package_v3.jsonl', 'as-what-same'),
    # ('./data/u2.jsonl', 'she-to-as'),
    ('./data/sat_package_v3.jsonl', 'what-is-to'),
    ('./data/u2.jsonl', 'what-is-to'),
]
lm = ('roberta-large', 32, 512)
export_dir = './experiments/ppl_pmi_negative/results'


def main():
    _model, _max_length, _batch = lm
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data, _temp in data:
        scorer.analogy_test(
            scoring_method='ppl_pmi',
            path_to_data=_data,
            template_types=[_temp],
            batch_size=_batch,
            export_dir=export_dir,
            permutation_negative=True,
            skip_scoring_prediction=True
        )
        scorer.release_cache()


if __name__ == '__main__':
    main()
