""" UnitTest for scorer """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import RelationScorer


class Test(unittest.TestCase):
    """Test"""

    def test(self):
        scorer = RelationScorer(model='roberta-base')

        batch_size = 4
        template_types = ['rel-same']

        path_to_data = './data/sample.jsonl'
        scorer.analogy_test(
            path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
            aggregation_positive='mean',
            permutation_negative=True,
            aggregation_negative='mean')

    def test_2(self):
        scorer = RelationScorer(model='roberta-base')

        batch_size = 4
        template_types = ['rel-same']

        path_to_data = './data/sample.jsonl'
        scorer.analogy_test(
            path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
            aggregation_positive='mean',
            permutation_negative=True,
            aggregation_negative='mean',
            skip_scoring_prediction=True
        )


if __name__ == "__main__":
    unittest.main()
