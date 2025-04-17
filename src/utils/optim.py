import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from ..models.layers.grn import GRN1d
from xlstm import xLSTMBlockStack

def configure_adamw(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
        """
        Adapted from karpathy's minGPT.
        Default values for parameters also taken from minGPT.
    
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.Conv1d, torch.nn.LSTM, GRN1d, xLSTMBlockStack)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif 'gamma' in pn and not isinstance(m, blacklist_weight_modules):
                    decay.add(fpn)
                elif 'beta' in pn and not isinstance(m, blacklist_weight_modules):
                    decay.add(fpn)
                elif 'learnable_skip' in pn or '_recurrent_kernel' in pn:
                    no_decay.add(fpn) # xlstm special params
                    

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer


def make_warmupCAWR(optimizer, warmup_steps, n_per_epoch, gamma=1., cycle_mult=1., min_lr=0.0):
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            WarmupLinearScheduleCosineDecay(optimizer, warmup_steps, t_total=n_per_epoch),
            CosineAnnealingWithDecay(optimizer, first_cycle_steps=int(1*n_per_epoch), gamma=gamma, cycle_mult=cycle_mult, min_lr=min_lr)
        ],
        milestones=[n_per_epoch]
        )


class WarmupLinearScheduleCosineDecay(LambdaLR):
    """ Linear warmup and then cosine decay.

        based on https://github.com/rishikksh20/CeiT-pytorch/blob/master/utils.py
        and https://github.com/TorAP/MemSeg/blob/main/scheduler.py

        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Cosine decays learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        if step > self.t_total:
            return 0.0
        cos_ = 0.5 * (1+math.cos(math.pi*float(step - self.warmup_steps) / float(max(1.0, self.t_total - self.warmup_steps))))
        return cos_


class CosineAnnealingWithDecay(LambdaLR):
    """
    via https://github.com/TorAP/MemSeg/blob/main/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 min_lr : float = 0.001,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = 1.
        self.max_lr = 1. # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, step):
        if step >= self.first_cycle_steps:
            if self.cycle_mult == 1.:
                self.step_in_cycle = (step-1) % self.first_cycle_steps
                self.cycle = max((step - 1), 0) // self.first_cycle_steps
            else:
                n = int(math.log((step / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                self.cycle = n
                self.step_in_cycle = step - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
        else:
            self.cur_cycle_steps = self.first_cycle_steps
            self.step_in_cycle = step
    
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        return self.max_lr * 0.5 * (1+math.cos(math.pi*float(self.step_in_cycle) / float(max(1.0, self.cur_cycle_steps))))
