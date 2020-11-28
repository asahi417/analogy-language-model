""" UnitTest for nested list """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import get_dataset_prompt
from alm.scoring_function import get_structure, restore_structure


class Test(unittest.TestCase):
    """Test"""

    def test(self):
        for i in ['./data/u2.jsonl', './data/u4.jsonl', './data/sat_package_v3.jsonl']:
            list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(i, None, True, True)

            # create batch
            batch_data, batch_id = get_structure(list_nested_sentence, batch_size=30)

            # run model prediction over flatten data
            list_score = batch_data.copy()

            # restore the nested structure
            list_placeholder = restore_structure(list_nested_sentence, list_score, batch_id)

            assert list_nested_sentence == list_placeholder


if __name__ == "__main__":
    unittest.main()
