import torch

# -*- coding:utf-8 -*-
"""
Created by 'zj'
"""
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


__all__ = ['PolyLR', 'StepLR', 'CosineLR']


"""
To use consistent api for scheduler, a new scheduler is created.

You can creat your own follow the code below.
 
----------------------------------------------------------------

Note that this learning rate decay strategy can not interact 
with other lr schedulers, because it use FIXed base_lr.

To interact with other strategy, use 
  [ group['lr'] * ... for group in self.optimizer.param_groups]
instead of
  [ base_lr * ... for base_lr in self.base_lrs]
"""


class WarmUpLRScheduler(object):
    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.iteration_per_epoch = iteration_per_epoch
        self.total_iteration = total_epoch * iteration_per_epoch
        self.warmup_epochs = warmup_epochs
        self.iteration_decay = iteration_decay

        # will not be changed
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

        self.step(0, 0)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def fget_lr(self, epoch, iter):
        raise NotImplementedError

    def step(self, epoch, iter):
        # decay with epoch
        if not self.iteration_decay:
            iter = 0

        # get normal iteration
        lr_list = self.get_lr(epoch, iter)

        # current iteration
        T = epoch * self.iteration_per_epoch + iter
        # warm up
        if self.warmup_epochs > 0 and T > 0 and epoch < self.warmup_epochs:
            # start from first iteration not 0
            lr_list = [lr * 1.0 * T /
                       (self.warmup_epochs*self.iteration_per_epoch) for lr in self.base_lrs]

        # adjust learning rate for all groups
        for param_group, lr in zip(self.optimizer.param_groups, lr_list):
            if 'lr_func' in param_group.keys():
                param_group['lr'] = param_group['lr_func'](lr)
            else:
                param_group['lr'] = lr
                
                
class PolyLR(WarmUpLRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    multiply by (1 - iter / total_iter) ** gamma.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        power (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        total_iter(int) : Total epoch
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, total_epoch, iteration_per_epoch, warmup_epochs=0, iteration_decay=True,
                 power=0.9):
        self.power = power
        super(PolyLR, self).__init__(optimizer, total_epoch, iteration_per_epoch, warmup_epochs, iteration_decay)

    def get_lr(self, epoch, iter):
        T = epoch * self.iteration_per_epoch + iter
        return [base_lr * ((1 - 1.0 * T / self.total_iteration) ** self.power)
                for base_lr in self.base_lrs]
                

class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, total_epoch, iteration_per_epoch, warmup_epochs=0, warmup_ratio=None, power=None, epoch=1):
        super().__init__(params, lr=lr, betas=betas,weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.total_epoch = total_epoch
        self.iteration_per_epoch = iteration_per_epoch
        self.total_iteration = total_epoch * iteration_per_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]
        self.step(0, 0)
        
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, epoch, iter):
        T = epoch * self.iteration_per_epoch + iter
        return [base_lr * ((1 - 1.0 * T / self.total_iteration) ** self.power)
                for base_lr in self.base_lrs]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter and epoch==1:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        else: 

            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1
        
        def step(self, epoch, iter):
        # decay with epoch
        if not self.iteration_decay:
            iter = 0

        # get normal iteration
        lr_list = self.get_lr(epoch, iter)

        # current iteration
        T = epoch * self.iteration_per_epoch + iter
        # warm up
        if self.warmup_epochs > 0 and T > 0 and epoch < self.warmup_epochs:
            # start from first iteration not 0
            lr_list = [lr * 1.0 * T /
                       (self.warmup_epochs*self.iteration_per_epoch) for lr in self.base_lrs]

        # adjust learning rate for all groups
        for param_group, lr in zip(self.optimizer.param_groups, lr_list):
            if 'lr_func' in param_group.keys():
                param_group['lr'] = param_group['lr_func'](lr)
            else:
                param_group['lr'] = lr