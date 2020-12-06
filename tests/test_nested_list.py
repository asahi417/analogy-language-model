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
            list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(i, None, True)

            # create batch
            batch_data_pos, batch_id_pos = get_structure(list_nested_sentence, batch_size=30, positive=True)
            batch_data_neg, batch_id_neg = get_structure(list_nested_sentence, batch_size=30, positive=False)

            # run model prediction over flatten data
            list_score_pos, list_score_neg = batch_data_pos.copy(), batch_data_neg.copy()

            # restore the nested structure
            list_placeholder = restore_structure(
                list_nested_sentence, list_score_pos, batch_id_pos, list_score_neg, batch_id_neg)

            assert list_nested_sentence == list_placeholder


if __name__ == "__main__":
    unittest.main()
