import torch


class Method(torch.nn.Module):
    """ Abstract class for methods.
    """
    def __init__(self, **kwargs):
        super(Method, self).__init__()

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def before_weight_update(self, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward is not implemented")

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError("Compute loss is not implemented")

    def visualize(self, *args, **kwargs):
        return {}

    @property
    def device(self):
        return next(self.parameters()).device
