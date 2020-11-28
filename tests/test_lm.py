""" UnitTest for LM modules """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import TransformersLM

MODEL = TransformersLM('roberta-base', max_length=128)
TEST = [
    "COVID-19 case numbers are rising rapidly across the whole of the UK and in other countries.",
    "US election: What a Biden presidency means for the UK"
]
TARGET = ['UK', 'Biden']


class Test(unittest.TestCase):
    """Test"""

    def test_pseudo_perplexity(self):
        ppl = MODEL.get_pseudo_perplexity(TEST)
        assert len(ppl) == len(TEST)

    def test_encode_plus_token_wise_mask(self):
        for n in range(len(TEST)):
            encode = MODEL.encode_plus_token_wise_mask(TEST[n])
            tokens = MODEL.tokenizer.tokenize(TEST[n])
            # should be same
            assert len(tokens) == len(encode)
            for i, t in zip(encode, tokens):
                input_ids = i['input_ids']
                mask_position = i['mask_position']
                mask_token_id = i['mask_token_id']
                # check if masked properly
                assert MODEL.tokenizer.mask_token == MODEL.tokenizer.convert_ids_to_tokens(input_ids[mask_position])
                masked_token = MODEL.tokenizer.convert_ids_to_tokens(mask_token_id)
                # should be masked in a sequential manner
                assert t == masked_token

    def test_get_log_likelihood(self):
        log_likelihood, (topk_prediction_values, topk_prediction_indices) = MODEL.get_nll(TEST, TARGET)
        logging.info(log_likelihood)
        logging.info(topk_prediction_values)
        logging.info(topk_prediction_indices)
        assert len(log_likelihood) == 2
        assert len(topk_prediction_indices) == 2
        assert len(topk_prediction_values) == 2
        log_likelihood, (topk_prediction_values, topk_prediction_indices) = MODEL.get_nll(TEST[0], TARGET[0])
        logging.info(log_likelihood)
        logging.info(topk_prediction_values)
        logging.info(topk_prediction_indices)
        assert len(log_likelihood) == 1
        assert len(topk_prediction_indices) == 1
        assert len(topk_prediction_values) == 1

    def test_encode_plus_mask(self):
        assert MODEL.sp_token_prefix == ['<s>']
        for padding in [True, False]:
            for i in range(len(TEST)):
                encode = MODEL.encode_plus_mask(TEST[i], TARGET[i], padding=padding)
                input_ids = encode['input_ids']
                if padding:
                    assert len(input_ids) == 128
                mask_position = encode['mask_position']
                mask_token_id = encode['mask_token_id']

                # masking position consistency
                assert MODEL.tokenizer.decode(mask_token_id).replace(' ', '') in TARGET[i]

                # check if masking is working
                assert MODEL.tokenizer.mask_token == MODEL.tokenizer.decode(input_ids[mask_position])

        # the case where the target token is split into subwords
        encode = MODEL.encode_plus_mask('shrubs are divided into subwords ', 'shrubs')
        input_ids = encode['input_ids']
        mask_position = encode['mask_position']
        mask_token_id = encode['mask_token_id']

        # masking position consistency
        assert MODEL.tokenizer.decode(mask_token_id) in 'shrubs'

        # check if masking is working
        assert MODEL.tokenizer.mask_token == MODEL.tokenizer.decode(input_ids[mask_position])


if __name__ == "__main__":
    unittest.main()
