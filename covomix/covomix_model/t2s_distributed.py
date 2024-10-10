import torch
from torch.autograd import Function
import torch.distributed as distributed

from einops import rearrange
from torch import Tensor, nn, einsum, IntTensor, LongTensor

import math
from pathlib import Path
from functools import partial
from random import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, nn, einsum, IntTensor, LongTensor

from torch.nn import Module, ModuleList

from torch.utils.data import Dataset

from einops import rearrange, repeat, pack, reduce
from einops.layers.torch import Rearrange


from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Union, Callable, Literal, Tuple, List

# distributed helpers
# types

FloatTensor = Union[
    torch.FloatTensor,
    torch.cuda.FloatTensor
]

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def empty(t: Tensor):
    return t.numel() == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        distributed.all_gather(sizes, size)
        sizes = torch.stack(sizes)

    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim = dim)

    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    distributed.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

class AllGather(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        is_dist = distributed.is_initialized() and distributed.get_world_size() > 1
        ctx.is_dist = is_dist

        if not is_dist:
            return x, None

        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        if not ctx.is_dist:
            return grads, None, None

        batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

all_gather = AllGather.apply