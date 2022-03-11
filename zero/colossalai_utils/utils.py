from common.utils import CONFIG, print_log
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from torch.distributed import get_rank


def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.logging import disable_existing_loggers
    from colossalai.nn.optimizer import CPUAdam
    # from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import TensorShardStrategy
    from colossalai.zero.sharded_model import ShardedModel, ShardedModelV2
    from colossalai.zero.sharded_optim import ShardedOptimizerV2

    disable_existing_loggers()
    colossalai.launch_from_torch(config=CONFIG)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    use_v2 = gpc.config.zero.pop('version', 2) == 2

    cpu_offload = gpc.config.zero.offload_config.device == 'cpu'

    rank = get_rank()
    reset_peak_memory_stats(rank)
    # if use_v2:
    shard_strategy = TensorShardStrategy()
    # with ZeroInitContext(convert_fp16='fp16' in gpc.config,
    #                      target_device=torch.device(gpc.config.zero.offload_config.device),
    #                      shard_strategy=shard_strategy,
    #                      shard_param=True):
    model = build_model()

    if use_v2:
        model = ShardedModelV2(model, shard_strategy, **gpc.config.zero)
    else:
        model = ShardedModel(model, **gpc.config.zero)

    criterion = build_loss()

    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')
    reset_peak_memory_stats(rank)

    if cpu_offload:
        optimizer = CPUAdam(model.parameters(),
                            lr=CONFIG['hyperparameter']['learning_rate'],
                            weight_decay=CONFIG['hyperparameter']['weight_decay'])
    else:
        optimizer = build_optimizer(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    if use_v2:
        optimizer = ShardedOptimizerV2(optimizer,
                                       model,
                                       shard_strategy,
                                       **gpc.config.get('fp16', dict()),
                                       cpu_offload=cpu_offload)

    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
