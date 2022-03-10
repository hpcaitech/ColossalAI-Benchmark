def init_w_col(builder, config):
    import colossalai
    from colossalai.amp import AMP_TYPE
    from colossalai.logging import disable_existing_loggers
    from colossalai.zero import ShardedModel

    col_amp = {'apex': AMP_TYPE.APEX, 'naive': AMP_TYPE.NAIVE, 'torch': AMP_TYPE.TORCH}
    if 'fp16' in config:
        config['fp16']['mode'] = col_amp[config['fp16']['mode']]

    disable_existing_loggers()
    colossalai.launch_from_torch(config=config)

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    model = build_model()
    model = ShardedModel(model, mixed_precision=True, reshard_after_forward=False, offload_config=dict(device='cpu'))

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    # scaler = ShardedGradScaler(**config['mixed_precision']) if 'mixed_precision' in config else None

    # engine, _, _, lr_scheduler = colossalai.initialize(model, optimizer, criterion, lr_scheduler=lr_scheduler)

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
