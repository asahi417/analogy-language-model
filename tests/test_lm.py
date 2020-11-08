""" UnitTest for TextRank """
import unittest
import logging
import re
from logging.config import dictConfig

from lm_util import TransformersLM

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
MODEL = TransformersLM('roberta-base')
TEST = "COVID-19 case numbers are rising rapidly across the whole of the UK and in other countries."


class TestTransformersLM(unittest.TestCase):
    """Test TransformersLM"""

    def test_encode_plus_with_mask(self):
        encode, mask_position, (masked_token, masked_token_id) = MODEL.encode_plus_with_mask(TEST, 'UK')

        # check consistency with encode
        input_ids = MODEL.tokenizer.encode(TEST)
        assert input_ids[mask_position] == masked_token_id

        # check consistency
        input_ids = encode['input_ids']
        input_ids[mask_position] = masked_token_id
        decoded_string = MODEL.tokenizer.decode(input_ids)
        assert len(re.findall(TEST, decoded_string)) > 0


if __name__ == "__main__":
    unittest.main()