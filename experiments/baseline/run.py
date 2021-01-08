import alm

all_templates = [['is-to-what'], ['is-to-as'], ['rel-same'], ['what-is-to'], ['she-to-as'], ['as-what-same']]
data = ['./data/sat_package_v3.jsonl', './data/u2.jsonl', './data/u4.jsonl']

export_dir = './experiments/baseline/results'


def main(lm):
    for _model, _max_length, _batch in lm:
        for scoring_method in ['ppl', 'embedding_similarity', 'ppl_pmi', 'pmi']:
            if scoring_method in ['pmi']:
                continue
            scorer = alm.RelationScorer(model=_model, max_length=_max_length)
            for _data in data:
                for _temp in all_templates:
                    if (_model == 'bert-large-cased' and scoring_method in ['ppl', 'embedding_similarity']) or \
                            (_model == 'gpt2-large' and scoring_method in ['ppl_pmi']):
                        scorer.analogy_test(
                            scoring_method=scoring_method,
                            path_to_data=_data,
                            template_types=_temp,
                            batch_size=_batch,
                            export_dir=export_dir,
                            permutation_negative=False,
                            skip_scoring_prediction=True
                        )
                        scorer.release_cache()

                # if 'gpt' not in _model:
                #     run('pmi')


if __name__ == '__main__':
    # main(lm=[('roberta-large', 32, 512), ('gpt2-xl', 32, 512)])
    main(lm=[('bert-large-cased', 64, 512), ('gpt2-large', 32, 512)])
