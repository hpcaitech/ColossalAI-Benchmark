import os

import torch
from torch.distributed import init_process_group
from common.utils import CONFIG


def init_w_fs(builder):
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.optim.grad_scaler import ShardedGradScaler
    from fairscale.nn.checkpoint import checkpoint_wrapper

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    init_process_group(rank=rank, world_size=world_size, init_method=f'tcp://{host}:{port}', backend='nccl')

    torch.cuda.set_device(rank)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    assert 'fsdp' in CONFIG
    use_checkpoint = CONFIG['model'].get('checkpoint')
    CONFIG['model']['checkpoint'] = False
    model = build_model()
    if use_checkpoint:
        model = checkpoint_wrapper(model)
    model = FSDP(model, **CONFIG['fsdp'])

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    scaler = ShardedGradScaler(**CONFIG['fp16']) if 'fp16' in CONFIG else None

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    return model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler
