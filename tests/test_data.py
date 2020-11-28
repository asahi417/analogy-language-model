""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import get_dataset, get_dataset_prompt


class Test(unittest.TestCase):
    """Test"""

    def test_dataset(self):
        tmp = get_dataset(path_to_data='./data/u2.jsonl')
        logging.info(tmp[:2])

    def test_dataset_prompt(self):
        _list_answer, _list_stem, _list_choice = None, None, None

        def show(_list_nested_sentence):
            list_options = _list_nested_sentence[0]
            logging.info("option_n: {}".format(len(list_options)))
            pos_list, neg_list = list_options[0]
            logging.info("positive ({}): \n{}".format(len(pos_list), pos_list))
            logging.info("negative ({}): \n{}".format(len(neg_list), neg_list))

        logging.info('\nno option')
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(
            path_to_data='./data/u2.jsonl',
            template_types=['is-to-what']
        )
        show(list_nested_sentence)

        logging.info('add template')
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(
            path_to_data='./data/u2.jsonl',
            template_types=['is-to-what', 'is-to-as']
        )
        show(list_nested_sentence)

        logging.info('add template + negative')
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(
            path_to_data='./data/u2.jsonl',
            permutation_negative=True,
            template_types=['is-to-what', 'is-to-as']
        )
        show(list_nested_sentence)

        logging.info('add template + positive')
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(
            path_to_data='./data/u2.jsonl',
            permutation_positive=True,
            template_types=['is-to-what', 'is-to-as']
        )
        show(list_nested_sentence)

        logging.info('add template + positive + negative')
        list_answer, list_nested_sentence, list_stem, list_choice = get_dataset_prompt(
            path_to_data='./data/u2.jsonl',
            permutation_positive=True,
            permutation_negative=True,
            template_types=['is-to-what', 'is-to-as']
        )
        show(list_nested_sentence)


if __name__ == "__main__":
    unittest.main()
