"""Minimal distributed helpers for evaluation."""

from __future__ import annotations

import os
import os.path as osp
import pickle
import shutil
from itertools import chain, zip_longest
from typing import Optional, Tuple

import torch
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup


def _init_dist_pytorch(backend, **kwargs) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group(backend=backend, **kwargs)


def is_distributed() -> bool:
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> Optional[ProcessGroup]:
    return torch_dist.distributed_c10d._get_default_group()


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    if not is_distributed():
        return 1
    return torch_dist.get_world_size(group or get_default_group())


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    if not is_distributed():
        return 0
    return torch_dist.get_rank(group or get_default_group())


def get_dist_info(group=None) -> Tuple[int, int]:
    return get_rank(group), get_world_size(group)


def barrier(group: Optional[ProcessGroup] = None) -> None:
    if is_distributed():
        torch_dist.barrier(group or get_default_group())


def collect_results_cpu(result_part: list, size: int, tmpdir="./dist_test_temp"):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    os.makedirs(tmpdir, exist_ok=True)
    with open(osp.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(result_part, f, protocol=2)

    barrier()
    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        path = osp.join(tmpdir, f"part_{i}.pkl")
        if not osp.exists(path):
            raise FileNotFoundError(f"Missing distributed result part: {path}")
        with open(path, "rb") as f:
            part_list.append(pickle.load(f))

    ordered_results = [item for item in chain.from_iterable(zip_longest(*part_list)) if item is not None]
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir, ignore_errors=True)
    return ordered_results
