import os

import torch
from torch.distributed import init_process_group


def init_w_fs(builder, config):
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.optim.grad_scaler import ShardedGradScaler

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    init_process_group(rank=rank, world_size=world_size, init_method=f'tcp://{host}:{port}', backend='nccl')

    torch.cuda.set_device(rank)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    assert 'fsdp' in config, 'No FSDP configuration provided.'
    model = build_model()
    model = FSDP(model, **config['fsdp'])

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    scaler = ShardedGradScaler(**config['mixed_precision']) if 'mixed_precision' in config else None

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    return model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler