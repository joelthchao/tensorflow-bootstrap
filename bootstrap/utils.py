import hashlib
import json
import os
from typing import Dict

import numpy as np


def make_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def pprint_params(params):
    exp_id = get_exp_id(params)
    print('Parameters {}:'.format(exp_id))
    print(json.dumps(params, indent=2, sort_keys=True))
    print()


def get_exp_id(params):
    params_str = json.dumps(params, indent=2, sort_keys=True)
    hash = hashlib.sha256(params_str.encode('utf-8')).hexdigest()
    return hash[:6]


def summarize_metrics(metrics_dict: Dict[str, list]):
    keys = sorted(metrics_dict.keys())
    strs = []
    for key in keys:
        mean_value = np.mean(metrics_dict[key])
        strs.append('{}: {}'.format(key, mean_value))

    return ' - '.join(strs)
