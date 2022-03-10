import torch
from common.utils import CONFIG


def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.logging import disable_existing_loggers
    from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import TensorShardStrategy
    from colossalai.zero.sharded_model import ShardedModel, ShardedModelV2
    from colossalai.zero.sharded_optim import ShardedOptimizerV2

    disable_existing_loggers()
    colossalai.launch_from_torch(config=CONFIG)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    use_v2 = gpc.config.get('sharded_model_version', 2) == 2

    if use_v2:
        shard_strategy = TensorShardStrategy()
        with ZeroInitContext(convert_fp16='fp16' in gpc.config,
                             target_device=torch.device(gpc.config.zero.offload_config.device),
                             shard_strategy=shard_strategy,
                             shard_param=True):
            model = build_model()
    else:
        model = build_model()

    if use_v2:
        model = ShardedModelV2(model, shard_strategy, **gpc.config.zero)
    else:
        model = ShardedModel(model, **gpc.config.zero)

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    cpu_offload = gpc.config.zero.offload_config.device == 'cpu'

    if use_v2:
        optimizer = ShardedOptimizerV2(optimizer, model, shard_strategy, **
                                       gpc.config.get('fp16', dict()), cpu_offload=cpu_offload)

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
