import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
import alm

all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
methods = ['ppl_tail_masked', 'ppl', 'embedding_similarity', 'ppl_pmi', 'pmi']

data = ['sat', 'u2', 'u4', 'google', 'bats']


def main(lm):
    for _model, _max_length, _batch in lm:
        for scoring_method in methods:
            if scoring_method in ['pmi'] and 'gpt' in _model:
                continue
            scorer = alm.RelationScorer(model=_model, max_length=_max_length)
            for _data in data:
                for _temp in all_templates:
                    scorer.analogy_test(
                        scoring_method=scoring_method,
                        data=_data,
                        template_type=_temp,
                        batch_size=_batch,
                        skip_scoring_prediction=True
                    )
                    scorer.release_cache()


if __name__ == '__main__':
    main(lm=[('roberta-large', 32, 512), ('gpt2-large', 32, 256), ('gpt2-xl', 32, 128), ('bert-large-cased', 64, 512)])
