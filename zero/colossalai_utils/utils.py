<<<<<<< HEAD
import torch
import torch.distributed as dist
from common.utils import CONFIG, print_log
=======
from common.utils import CONFIG
>>>>>>> b90b3ab... rollbacked flops profiler to hardcoding; reworked how cpu adam is used for different methods


def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.logging import disable_existing_loggers
    # from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import TensorShardStrategy
    from colossalai.zero.sharded_model import ShardedModel, ShardedModelV2
    from colossalai.zero.sharded_optim import ShardedOptimizerV2
    from colossalai.nn.optimizer import CPUAdam

    disable_existing_loggers()
    colossalai.launch_from_torch(config=CONFIG)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    use_v2 = gpc.config.zero.pop('version', 2) == 2

<<<<<<< HEAD
    print_log('Building model')
    if use_v2:
        shard_strategy = TensorShardStrategy()
        # with ZeroInitContext(convert_fp16='fp16' in gpc.config,
        #                      target_device=torch.device(gpc.config.zero.offload_config.device),
        #                      shard_strategy=shard_strategy,
        #                      shard_param=True):
        #     model = build_model()
        model = build_model()
    else:
        model = build_model()
=======
    cpu_offload = gpc.config.zero.offload_config.device == 'cpu'

    # if use_v2:
    shard_strategy = TensorShardStrategy()
    # with ZeroInitContext(convert_fp16='fp16' in gpc.config,
    #                      target_device=torch.device(gpc.config.zero.offload_config.device),
    #                      shard_strategy=shard_strategy,
    #                      shard_param=True):
    model = build_model()
>>>>>>> b90b3ab... rollbacked flops profiler to hardcoding; reworked how cpu adam is used for different methods

    if use_v2:
        model = ShardedModelV2(model, shard_strategy, **gpc.config.zero)
    else:
        model = ShardedModel(model, **gpc.config.zero)

    criterion = build_loss()
<<<<<<< HEAD
    print_log(
        f'GPU Mem: {torch.cuda.max_memory_allocated(dist.get_rank()) / (1024 * 1024)} M')
    print_log('Building optimizer')
    optimizer = build_optimizer(model.parameters())
=======

    if cpu_offload:
        optimizer = CPUAdam(model.parameters(),
                            lr=CONFIG['hyperparameter']['learning_rate'],
                            weight_decay=CONFIG['hyperparameter']['weight_decay'])
    else:
        optimizer = build_optimizer(model.parameters())
>>>>>>> b90b3ab... rollbacked flops profiler to hardcoding; reworked how cpu adam is used for different methods

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    if use_v2:
        optimizer = ShardedOptimizerV2(optimizer,
                                       model,
                                       shard_strategy,
                                       **gpc.config.get('fp16', dict()),
                                       cpu_offload=cpu_offload)

    print_log(
        f'GPU Mem: {torch.cuda.max_memory_allocated(dist.get_rank()) / (1024 * 1024)} M')
    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
