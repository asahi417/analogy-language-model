""" UnitTest for LM modules """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import TransformersLM

TEST = [
    "The COVID-19 case numbers are rising rapidly across the whole of the UK and in other countries.",
    "US election: What a Biden presidency means for the UK"
]
TARGET = [['UK', 'COVID-19'], ['Biden']]


def test(model):
    ppl = model.get_perplexity(TEST)
    logging.info('`get_perplexity`: {}'.format(ppl))
    assert len(ppl) == len(TEST)


class Test(unittest.TestCase):
    """Test"""

    # def test_lm(self):
    #     logging.info('test LM (gpt2)')
    #     model = TransformersLM('gpt2', max_length=32)
    #     test(model)

    def test_mlm(self):
        logging.info('test MLM (albert-base-v1)')
        model = TransformersLM('albert-base-v1', max_length=32)
        test(model)

        log_likelihood = model.get_nll(TEST, TARGET)
        logging.info('`get_nll`: {}'.format(log_likelihood))
        assert len(log_likelihood) == len(TEST)


if __name__ == "__main__":
    unittest.main()
