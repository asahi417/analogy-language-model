""" UnitTest for scorer """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import RelationScorer


class Test(unittest.TestCase):
    """Test"""

    def test(self):
        data = './data/u2.jsonl'
        temps = ['is-to-what', 'is-to-as']
        # temps = ['is-to-what']
        scorer = RelationScorer()
        # scorer.analogy_test(data, template_types=temps)
        scorer.analogy_test(data, template_types=temps, aggregation_positive='max')


if __name__ == "__main__":
    unittest.main()
