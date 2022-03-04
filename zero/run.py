import json
import os
import sys

from colossalai_utils.utils import init_w_col
from deepspeed_utils.utils import init_w_ds
from fairscale_utils.utils import init_w_fs
from common.gpt import gpt_builder
from common.train import train, _print_log
from patrickstar_utils.utils import init_w_ps
from torch_utils.utils import init_w_torch

_zero_method = {
    'fairscale': init_w_fs,
    'colossalai': init_w_col,
    'torch': init_w_torch,
    'patrickstar': init_w_ps,
    'deepspeed': init_w_ds
}


def run_zero(config):
    method = config['method']
    assert method in ['colossalai', 'deepspeed', 'fairscale', 'patrickstar', 'torch'], f'No support for {method}.'

    train(*_zero_method[method](gpt_builder, config), config=config)


if __name__ == '__main__':
    config_file = None
    if sys.argv[1] == '--config':
        config_file = sys.argv[2]
        sys.argv = sys.argv[2:]
    elif sys.argv[1].startswith('--config='):
        config_file = sys.argv[1][8:]
        sys.argv = sys.argv[1:]
    else:
        raise ValueError('No valid config file found.')

    assert os.path.exists(config_file), 'No valid config file found.'
    with open(config_file, 'r') as f:
        config = json.load(f)

    _print_log(f'Initializing {config["method"]} ...')

    run_zero(config)
