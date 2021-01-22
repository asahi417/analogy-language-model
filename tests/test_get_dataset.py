""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import AnalogyData


class Test(unittest.TestCase):
    """Test"""

    def test_dataset_test(self):
        for i in [False, True]:
            data_instance = AnalogyData(test=False, data='sat', negative_permutation=i)
            logging.info(data_instance.flatten_pos[:10])
            input()
            out = data_instance.insert_score(data_instance.flatten_pos, data_instance.flatten_neg)
            logging.info(out[:10])
            # logging.info(data_instance.list_nested_permutation)
            assert out == data_instance.list_nested_permutation, i

    # def test_check_relation_token_uniqueness(self):
    #     for v in [True, False]:
    #         logging.info(v)
    #         data_instance = AnalogyData(
    #             data_name='u2', template_types=['is-to-as'], permutation_negative=False, validation=v)
    #         prompt, relation = data_instance.get_prompt()
    #         for i in relation:
    #             assert len(list(set(i))) == 4, i
    #
    #         data_instance = AnalogyData(
    #             data_name='u4', template_types=['as-what-same'], permutation_negative=False, validation=v)
    #         prompt, relation = data_instance.get_prompt()
    #         for i in relation:
    #             assert len(list(set(i))) == 4, i
    #
    #         data_instance = AnalogyData(
    #             data_name='sat', template_types=['is-to-as'], permutation_negative=False, validation=v)
    #         prompt, relation = data_instance.get_prompt()
    #         for i in relation:
    #             assert len(list(set(i))) == 4, i
    #
    #         data_instance = AnalogyData(
    #             data_name='bats', template_types=['is-to-as'], permutation_negative=False, validation=v)
    #         prompt, relation = data_instance.get_prompt()
    #         for i in relation:
    #             assert len(list(set(i))) == 4, i
    #
    #         data_instance = AnalogyData(
    #             data_name='google', template_types=['is-to-as'], permutation_negative=False, validation=v)
    #         prompt, relation = data_instance.get_prompt()
    #         for i in relation:
    #             assert len(list(set(i))) == 4, i



if __name__ == "__main__":
    unittest.main()
