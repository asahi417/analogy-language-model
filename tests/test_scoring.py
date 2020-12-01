""" UnitTest for scorer """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import RelationScorer


class Test(unittest.TestCase):
    """Test"""

    def test(self):
        data = './data/sample.jsonl'
        temps = ['is-to-what', 'is-to-as']
        # temps = ['is-to-what']
        scorer = RelationScorer()
        # scorer.analogy_test(data, template_types=temps)
        scorer.analogy_test(data, template_types=temps, aggregation_positive='max')

    # def test_2(self):
    #     scorer = RelationScorer(model='roberta-large')
    #
    #     batch_size = 16
    #     template_types = ['rel-same']
    #
    #     path_to_data = './data/sat_package_v3-0.jsonl'
    #     scorer.analogy_test(
    #         path_to_data=path_to_data, template_types=template_types, batch_size=batch_size,
    #         permutation_positive=False, aggregation_positive='none',
    #         permutation_negative=True, aggregation_negative='mean')


if __name__ == "__main__":
    unittest.main()
