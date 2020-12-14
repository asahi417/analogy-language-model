""" UnitTest for scorer """
import unittest
import logging
import shutil
import os

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
if os.path.exists('./tests/results'):
    shutil.rmtree('./tests/results')

from alm import RelationScorer
scorer = RelationScorer('albert-base-v1', max_length=32)


class Test(unittest.TestCase):
    """Test"""

    # def test_ppl(self):
    #     s = scorer.analogy_test(
    #         export_dir='./tests/results',
    #         scoring_method='ppl',
    #         path_to_data='./data/sample.jsonl',
    #         template_types=['rel-same'],
    #         batch_size=4,
    #         aggregation_positive='mean')
    #     logging.info(s)
    #
    #     # negative
    #     s = scorer.analogy_test(
    #         export_dir='./tests/results',
    #         scoring_method='ppl',
    #         path_to_data='./data/sample.jsonl',
    #         template_types=['rel-same'],
    #         batch_size=4,
    #         aggregation_positive='mean',
    #         permutation_negative=True,
    #         aggregation_negative='mean')
    #     logging.info(s)

    def test_pmi(self):
        s = scorer.analogy_test(
            export_dir='./tests/results',
            scoring_method='pmi',
            scoring_method_config={"aggregation": 0},
            path_to_data='./data/sample.jsonl',
            template_types=['rel-same'],
            batch_size=4,
            aggregation_positive='mean')
        logging.info(s)

        s = scorer.analogy_test(
            export_dir='./tests/results',
            scoring_method='pmi',
            scoring_method_config={"aggregation": 11},
            path_to_data='./data/sample.jsonl',
            template_types=['rel-same'],
            batch_size=4,
            aggregation_positive='mean')
        logging.info(s)

        s = scorer.analogy_test(
            export_dir='./tests/results',
            scoring_method='pmi',
            scoring_method_config={"aggregation": 'mean'},
            path_to_data='./data/sample.jsonl',
            template_types=['rel-same'],
            batch_size=4,
            aggregation_positive='mean')
        logging.info(s)

        # negative
        s = scorer.analogy_test(
            export_dir='./tests/results',
            scoring_method='pmi',
            scoring_method_config={"aggregation": 0},
            path_to_data='./data/sample.jsonl',
            template_types=['rel-same'],
            batch_size=4,
            aggregation_positive='mean',
            permutation_negative=True,
            aggregation_negative='mean')
        logging.info(s)

        s = scorer.analogy_test(
            export_dir='./tests/results',
            scoring_method='pmi',
            scoring_method_config={"aggregation": 11},
            path_to_data='./data/sample.jsonl',
            template_types=['rel-same'],
            batch_size=4,
            aggregation_positive='mean',
            permutation_negative=True,
            aggregation_negative='mean')
        logging.info(s)

        s = scorer.analogy_test(
            export_dir='./tests/results',
            scoring_method='pmi',
            scoring_method_config={"aggregation": 'mean'},
            path_to_data='./data/sample.jsonl',
            template_types=['rel-same'],
            batch_size=4,
            aggregation_positive='mean',
            permutation_negative=True,
            aggregation_negative='mean')
        logging.info(s)


if __name__ == "__main__":
    unittest.main()
