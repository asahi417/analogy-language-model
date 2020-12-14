""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import AnalogyData


class Test(unittest.TestCase):
    """Test"""

    def test_dataset(self):
        data_instance = AnalogyData(
            path_to_data='./data/sample.jsonl', template_types=['is-to-as'], permutation_negative=False)
        logging.info(data_instance.flatten_prompt_pos)
        prompt, relation = data_instance.get_prompt(return_relation_pairs=True)
        logging.info(prompt)
        tmp_score = list(zip(prompt, relation))
        out = data_instance.insert_score(tmp_score)
        logging.info(out)
        logging.info(data_instance.list_nested_sentence)
        assert out == data_instance.list_nested_sentence


if __name__ == "__main__":
    unittest.main()
