import torch
from zero.common.utils import CONFIG, print_log
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from torch.distributed import get_rank

def init_w_col(builder):
    import colossalai
    from colossalai.core import global_context as gpc
    from colossalai.nn.optimizer import HybridAdam
    from colossalai.zero.init_ctx import ZeroInitContext
    from colossalai.zero.shard_utils import (BucketTensorShardStrategy)

    from colossalai.utils import colo_set_process_memory_fraction

    from colossalai.logging import disable_existing_loggers
    disable_existing_loggers()

    colo_set_process_memory_fraction(1.0)

    colossalai.launch_from_torch(config=CONFIG)

    build_data, build_model, build_loss, optimizer_class, build_scheduler = builder()

    print_log('Building data')
    train_data, test_data = build_data()

    use_zero = "zero" in gpc.config
    if use_zero:
        cpu_offload = gpc.config.zero.model_config.tensor_placement_policy == 'cpu'
    else:
        cpu_offload = None

    rank = get_rank()
    reset_peak_memory_stats(rank)

    print_log('Building model')
    if use_zero:
        shard_strategy = BucketTensorShardStrategy()
        with ZeroInitContext(target_device=torch.cuda.current_device(),
                            shard_strategy=shard_strategy,
                            shard_param=True):
            model = build_model()
        gpc.config.zero.model_config['shard_strategy'] = shard_strategy

    else:
        model = build_model()

    criterion = build_loss()

    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')
    reset_peak_memory_stats(rank)

    optimizer_class = HybridAdam
    optimizer_kwargs = {
        'lr': CONFIG['hyperparameter']['learning_rate'],
        'weight_decay': CONFIG['hyperparameter']['weight_decay']
    }

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    lr_scheduler = build_scheduler(len(train_data), optimizer)
    print_log(f'Peak Memory = {max_memory_allocated(rank) / (1024 * 1024)} M')

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
