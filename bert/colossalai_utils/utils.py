import torch
from zero.common.utils import CONFIG, print_log
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from torch.distributed import get_rank

def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.nn.optimizer import CPUAdam
    from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import (BucketTensorShardStrategy,
                                             TensorShardStrategy)
    from colossalai.zero.sharded_model import ShardedModelV2
    from colossalai.zero.sharded_optim import ShardedOptimizerV2

    colossalai.launch_from_torch(config=CONFIG)

    build_data, build_model, build_loss, optimizer_class, build_scheduler = builder()

    print_log('Building data')
    train_data, test_data = build_data()

    use_zero = "zero" in gpc.config
    if use_zero:
        use_v2 = gpc.config.zero.pop('version', 2) == 2
        cpu_offload = gpc.config.zero.offload_config.device == 'cpu'
    else:
        use_v2 = False
        cpu_offload = None

    rank = get_rank()
    reset_peak_memory_stats(rank)

    print_log('Building model')
    if use_zero:
        shard_strategy = BucketTensorShardStrategy()
        with ZeroInitContext(convert_fp16='fp16' in gpc.config,
                            target_device=torch.cuda.current_device(),
                            shard_strategy=shard_strategy,
                            shard_param=True):
            model = build_model()
        model = ShardedModelV2(model, shard_strategy, **gpc.config.zero)

    else:
        model = build_model()

    criterion = build_loss()

    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')
    reset_peak_memory_stats(rank)

    optimizer_kwargs = {}
    if use_zero and cpu_offload:
        optimizer_class = CPUAdam
        optimizer_kwargs = {
            'lr': CONFIG['hyperparameter']['learning_rate'],
            'weight_decay': CONFIG['hyperparameter']['weight_decay']
        }

    if use_zero:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        optimizer = ShardedOptimizerV2(model,
                                       optimizer,
                                       **gpc.config.get('fp16', dict()),
                                       cpu_offload=cpu_offload)
    else:
        optimizer = optimizer_class(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)
    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')

    if not use_zero:
        engine, train_data, test_data, lr_scheduler = colossalai.initialize(model, 
                                                                        optimizer, 
                                                                        criterion, 
                                                                        train_data, 
                                                                        test_data,
                                                                        lr_scheduler)
        model = engine
        criterion = engine.criterion
        optimizer = engine

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
