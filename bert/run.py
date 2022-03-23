import sys
sys.path.append('../zero/')

import os

from src.bert import bert_builder
from common.train import train
from common.utils import CONFIG, load_config, print_log
from torch_utils.utils import init_w_torch

_method = {
    'torch': init_w_torch,
}

_builder = {
    'bert': bert_builder,
}

def run_bert():
    method = CONFIG['method']

    model = CONFIG['model']['type']
    model_type = model.split('_')[0]

    train(*_method[method](_builder[model_type]))

if __name__ == '__main__':
    load_config()

    CONFIG['log_path'] = os.environ.get('LOG', '.')
    os.makedirs(CONFIG['log_path'], exist_ok=True)

    print_log(f'Initializing {CONFIG["method"]} ...')

    run_bert()
