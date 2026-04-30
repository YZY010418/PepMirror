#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple
from functools import wraps

import torch
import torch.distributed as dist
from utils.logger import print_log


OOMReturn = namedtuple('OOMReturn', ['fake_loss'])


def oom_decorator(forward):
    @wraps(forward)

    def deco_func(self, *args, **kwargs):
        try:
            output = forward(self, *args, **kwargs)
            return output
        except RuntimeError as e:
            if 'out of memory' in str(e):
                output = sum([p.norm() for p in self.parameters() if p.dtype == torch.float]) * 0.0
                return OOMReturn(output)
            else:
                raise e
    
    return deco_func

def _ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def _any_rank_invalid(loss: torch.Tensor) -> bool:
    finite = torch.isfinite(loss)
    if finite.ndim > 0:
        finite = finite.all()
    invalid_local = (~finite).to(torch.int32)
    if _ddp_is_initialized():
        dist.all_reduce(invalid_local, op=dist.ReduceOp.MAX)
    return invalid_local.item() != 0

def safe_backward(loss, model):
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    if _any_rank_invalid(loss):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        torch.cuda.empty_cache()
        print_log(f'invalid loss, skip', level='WARN')
        return False 

    try:
        loss.backward()
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            dummy = sum(p.detach().float().norm() for p in model.parameters()
                        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16, torch.float32)) * 0.0
            dummy.backward()
            torch.cuda.empty_cache()
            print_log(f'Backward out of memory, skip', level='WARN')
            return False
        else:
            raise
