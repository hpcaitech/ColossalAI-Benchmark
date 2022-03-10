import os

from colossalai_utils.utils import init_w_col
from common.gpt2 import gpt2_builder
from common.train import train
from common.utils import CONFIG, load_config, print_log
from common.vit import vit_builder
from deepspeed_utils.utils import init_w_ds
from fairscale_utils.utils import init_w_fs
from patrickstar_utils.utils import init_w_ps
from torch_utils.utils import init_w_torch

_zero_method = {
    'fairscale': init_w_fs,
    'colossalai': init_w_col,
    'torch': init_w_torch,
    'patrickstar': init_w_ps,
    'deepspeed': init_w_ds
}

_builder = {
    'gpt2': gpt2_builder,
    'vit': vit_builder,
}


def run_zero():
    method = CONFIG['method']
    assert method in ['colossalai', 'deepspeed', 'fairscale', 'patrickstar', 'torch'], f'No support for {method}.'

    model = CONFIG['model']['type']
    model_type = model.split('_')[0]
    assert model_type in ['gpt2', 'vit'], f'No support for {model}.'

    train(*_zero_method[method](_builder[model_type]))


if __name__ == '__main__':
    load_config()

    CONFIG['log_path'] = os.environ.get('LOG', '.')
    os.makedirs(CONFIG['log_path'], exist_ok=True)

    print_log(f'Initializing {CONFIG["method"]} ...')

    run_zero()
