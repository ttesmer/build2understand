import torch
from torch import matmul, cat, sigmoid, tanh
from collections import OrderedDict

class SGD:
    def __init__(self, model, lr, weight_decay=0, maximize=False):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.maximize = maximize

        if not model.forward:
            raise Exception("`model` given to SGD has no method `forward`.")
        if not model.backward:

    def step(self):
        model = self.model
        lr = self.lr
        weight_decay = self.weight_decay
        maximize = self.maximize

        params = model.params
        dparams = model.dparams

        # TODO: add momentum
        # - (https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD.step)
        # - what is `momentum_buffer_list`? 
        for param, dparam in zip(params, dparams):

            if weight_decay != 0:
                dparam = dparam.add(param, alpha=weight_decay)

            alpha = lr if maximize else -lr
            param.add_(dparam, alpha=alpha)
