""" Dictionary structure keeper """
from copy import deepcopy
from typing import Dict, List
from itertools import chain

__all__ = "DictKeeper"


def flatten_dict(dict_structure, target_key: str = 'encode'):

    def _flatten_dict(_dict, _encode: List = None):
        """ flatten nested list """
        _encode = [] if _encode is None else _encode
        if target_key in _dict.keys():
            _encode += [_dict.pop(target_key)]
            return _flatten_dict(_dict, _encode)
        elif 'child' in _dict.keys():
            _encode += list(chain(*[
                _flatten_dict(_dict['child'][k])
                for k in sorted(_dict['child'].keys())]))
            return _encode
        else:
            return _encode

    return _flatten_dict(deepcopy(dict_structure))


def replace_dict(dict_structure, insert_values: List, insert_key: str = 'score', replace_key: str = 'encode'):

    def replace_dict_value(_dict, values: List):
        """ flatten nested list """
        if replace_key in _dict.keys():
            _dict.pop(replace_key)
            _dict[insert_key] = values.pop(0)
            return replace_dict_value(_dict, values)
        elif 'child' in _dict.keys():
            for k in sorted(_dict['child'].keys()):
                replace_dict_value(_dict['child'][k], values)
            return _dict
        else:
            return _dict

    return replace_dict_value(deepcopy(dict_structure), deepcopy(insert_values))


class DictKeeper:
    """ keep a nested dict structure and replace the value with flatten values, restoring original nest """

    def __init__(self, _dict: Dict, target_key: str = 'encode'):
        print(_dict)
        self.original_dict = _dict
        self.target_key = target_key
        self.flat_values = flatten_dict(_dict, target_key=self.target_key)

    def restore_structure(self, flat_values, insert_key: str = 'score'):
        assert len(flat_values) == len(self.flat_values), 'inconsistent length: {} != {}'.format(
            len(flat_values), len(self.flat_values))

        return replace_dict(self.original_dict, flat_values, insert_key=insert_key, replace_key=self.target_key)

    def __len__(self):
        return len(self.flat_values)









