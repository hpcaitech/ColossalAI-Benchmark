import torch
from common.utils import CONFIG


def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.logging import disable_existing_loggers
    from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import TensorShardStrategy
    from colossalai.zero.sharded_model import ShardedModelV2
    from colossalai.zero.sharded_optim import ShardedOptimizerV2

    disable_existing_loggers()
    colossalai.launch_from_torch(config=CONFIG)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    shard_strategy = TensorShardStrategy()
    with ZeroInitContext(convert_fp16='fp16' in gpc.config,
                         convert_cuda=torch.cuda.is_available(),
                         shard_strategy=shard_strategy,
                         shard_param=True):
        model = build_model()
    model = ShardedModelV2(model, shard_strategy, **gpc.config.zero)

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    optimizer = ShardedOptimizerV2(optimizer, model, shard_strategy, **gpc.config.get('fp16', dict()))

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
