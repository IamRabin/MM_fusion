# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import socket

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Let the OS assign a free port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def init_distributed(port=37123, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    for _ in range(5):  # Try up to 5 ports before giving up
        try:
            os.environ['MASTER_PORT'] = str(port)
            
            # only for single node setups
            store = torch.distributed.FileStore("tmp_store", world_size)

            torch.distributed.init_process_group(
                backend='nccl',
                world_size=world_size,
                # store=store,
                rank=rank
            )
            break  # Successfully initialized, break the loop
        except RuntimeError as e:
            if "Address already in use" in str(e):
                logger.info(f"Port {port} is in use, trying to find a free port.")
                port = find_free_port()  # Find a new port
            else:
                raise e  # Reraise if it's a different error
        except Exception as e:
            world_size, rank = 1, 0
            logger.info(f'Rank: {rank}. Distributed training not available {e}')
            break

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
