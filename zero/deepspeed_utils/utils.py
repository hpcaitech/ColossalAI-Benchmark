def init_w_ds(builder):
    import deepspeed
    from common.utils import CONFIG

    config = CONFIG

    deepspeed.init_distributed()

    build_data, build_model, build_loss, build_optimizer, build_scheduler = builder()

    train_data, test_data = build_data()

    with deepspeed.zero.Init(config_dict_or_path=config):
        model = build_model()

    criterion = build_loss()

    optimizer = build_optimizer(model.parameters())

    lr_scheduler = build_scheduler(len(train_data), optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model,
                                                             optimizer=optimizer,
                                                             lr_scheduler=lr_scheduler,
                                                             config=config)

    return model, train_data, test_data, criterion, optimizer, None, lr_scheduler
