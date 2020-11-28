""" UnitTest for prompting modules """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from alm import prompting_relation


class Test(unittest.TestCase):
    """Test"""

    def test_prompting(self):
        sample = {
            "stem": ["arid", "dry"],
            "answer": 0,
            "choice": [
                ["glacial", "cold"], ["coastal", "tidal"], ["damp", "muddy"], ["snowbound", "polar"],
                ["shallow", "deep"]],
            "prefix": "190 FROM REAL SATs"}
        prompt = prompting_relation(
            subject_stem=sample['stem'][0],
            object_stem=sample['stem'][1],
            subject_analogy=sample['choice'][0][1],
            object_analogy=sample['choice'][0][1])
        logging.info(prompt)


if __name__ == "__main__":
    unittest.main()
