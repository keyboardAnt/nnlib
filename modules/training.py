from torch import optim


def build_optimizer(params, training_args):
    args = training_args['optimizer']
    optimizer = None
    name = args.pop('name', 'adam')
    if name == 'adam':
        optimizer = optim.Adam(params, **args)
    if name == 'sgd':
        optimizer = optim.SGD(params, **args)
    return optimizer


def build_scheduler(optimizer, training_args):
    args = training_args.get('scheduler', {})
    step_size = args.get('step_size', 1)
    gamma = args.get('gamma', 1.0)
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
