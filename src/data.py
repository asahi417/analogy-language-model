import json
import os

__all__ = 'get_dataset'


def get_dataset(path_to_data: str):
    """ get prompted SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    assert os.path.exists(path_to_data)

    with open(path_to_data, 'r') as f:
        return list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
