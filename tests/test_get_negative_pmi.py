""" UnitTest for LM encoding """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import TransformersLM, DictKeeper

TEST = [
    "The COVID19 case numbers are rising rapidly across the whole of the UK and in other countries.",
    "US election: What a Biden presidency means for the UK",
    # "The relation between stay and depart is the same as the relation between take and move"
]
TARGET = ['COVID19', 'UK']
TARGET_MASK = ['UK', 'US election']
model = TransformersLM('albert-base-v1', max_length=32)


class Test(unittest.TestCase):
    """Test"""

    def test_dict_keeper(self):
        for i in range(2):
            dict_keeper = model.encode_plus_mask(TEST[i], TARGET[i], TARGET_MASK[i])
            original_encode = dict_keeper.original_dict

            tmp_dict = dict_keeper.restore_structure(dict_keeper.flat_values, insert_key='tmp')
            dict_keeper = DictKeeper(tmp_dict, target_key='tmp')
            tmp_dict = dict_keeper.restore_structure(dict_keeper.flat_values, insert_key='encode')

            assert tmp_dict == original_encode

    def test_mlm(self):
        score = model.get_negative_pmi(TEST, tokens_to_mask=TARGET)
        assert len(score) == len(TEST)
        logging.info(score)

        score = model.get_negative_pmi(TEST, tokens_to_mask=TARGET, tokens_to_condition=TARGET_MASK)
        assert len(score) == len(TEST)
        logging.info(score)

    def test_token_position(self):
        for i in range(2):
            token = model.tokenizer.tokenize(TEST[i])
            s, e = model.find_position(str_to_mask=TARGET[i], text=TEST[i], token=token)
            found_word = model.tokenizer.convert_tokens_to_string(token[s:e])
            logging.info(' - target: {}'.format(TARGET[i]))
            logging.info(' - found: {}'.format(found_word))
            assert found_word in TARGET[i] or found_word in TARGET[i].lower()



if __name__ == "__main__":
    unittest.main()
