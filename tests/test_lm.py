""" UnitTest for LM modules """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import TransformersLM

MODEL = TransformersLM('roberta-base', max_length=128)
TEST = [
    "The COVID-19 case numbers are rising rapidly across the whole of the UK and in other countries.",
    "US election: What a Biden presidency means for the UK"
]
TARGET = [['UK', 'COVID-19'], ['Biden']]


class Test(unittest.TestCase):
    """Test"""

    def test_get_perplexity(self):
        ppl = MODEL.get_perplexity(TEST)
        logging.info('`get_perplexity`: {}'.format(ppl))
        assert len(ppl) == len(TEST)

    def test_nll(self):
        log_likelihood = MODEL.get_nll(TEST, TARGET)
        logging.info(log_likelihood)
        assert len(log_likelihood) == len(TEST)


if __name__ == "__main__":
    unittest.main()
