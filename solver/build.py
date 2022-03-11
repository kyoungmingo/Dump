from torch.optim import SGD, Adam


def make_optimizer(args, model):
    params = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay

        params += [
            {"params": [v], "lr": lr, "initial_lr": lr, "weight_decay": weight_decay}
        ]

    if args.optimizer == "sgd":
        optimizer = SGD(params, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = Adam(params)
    else:
        raise NotImplementedError()

    return optimizer
