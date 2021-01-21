import alm

templates = [['what-is-to']]

data = ['./data/google.jsonl', './data/bats.jsonl']
lm = ('roberta-large', 32, 512)
export_dir = './experiments/relation_dataset/results'


def main():
    _model, _max_length, _batch = lm
    scorer = alm.RelationScorer(model=_model, max_length=_max_length)
    for _data in data:
        for _temp in templates:
            scorer.analogy_test(
                scoring_method='ppl_pmi',
                path_to_data=_data,
                template_types=_temp,
                batch_size=_batch,
                export_dir=export_dir,
                permutation_negative=True,
                skip_scoring_prediction=True
            )
            scorer.release_cache()


if __name__ == '__main__':
    main()
