#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context

import inspect
import random
import socket
import sys
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist

from utils.config import Config

from . import process_group_initializer as pgroup_initializer
from .process_group_initializer import ParallelMode

IS_TENSOR_PARALLEL = "is_tensor_parallel"
IS_SEQUENCE_PARALLEL = "is_sequence_parallel"
IS_REPLICA_ZERO_PARALLEL = "is_replica_zero_parallel"
IS_SEQUENCE_DATA_PARALLEL = "is_sequence_data_parallel"
IS_WEIGHT_ZERO_PARALLEL = "is_weight_zero_parallel"


class ParallelContext:
    """This class provides interface functions for users to get the parallel context,
    such as the global rank, the local rank, the world size, etc. of each device.

    """

    def __init__(self):
        # distributed settings
        self._global_ranks = dict()
        self._local_ranks = dict()
        self._world_sizes = dict()
        self._groups = dict()
        self._cpu_groups = dict()
        self._ranks_in_group = dict()

        # load config from file
        self._config = None

        # default parallel args, will be overwritten during process group intialization
        self.world_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.zero1_parallel_size = -1
        self.nettest_parallel_size = 1
        self.expert_parallel_size = -1
        self.num_processes_on_current_node = -1
        self.virtual_pipeline_parallel_size = None
        self.virtual_pipeline_parallel_rank = None
        self._expert_parallel_group_names = []

    @property
    def config(self):
        return self._config

    @property
    def expert_parallel_group_names(self):
        return self._expert_parallel_group_names

    def load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dict or a file.

        Args:
            config (dict or str): Either a dict containing the configuration information or the filename
                of a file containing the configuration information.

        Raises:
            TypeError: Raises a TypeError if `config` is neither a dict nor a str.
        """
        if isinstance(config, str):
            self._config = Config.from_file(config)
        elif isinstance(config, dict):
            self._config = Config(config)
        else:
            raise TypeError("Invalid type for config, only dictionary or string is supported")

    def detect_num_processes_on_current_node(self):
        hostname = socket.gethostname()
        hostname_list = [None for _ in range(self.get_world_size(ParallelMode.GLOBAL))]
        dist.all_gather_object(hostname_list, hostname, group=self.get_group(ParallelMode.GLOBAL))
        counter = Counter(hostname_list)
        self.num_processes_on_current_node = counter[hostname]

    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        assert isinstance(
            parallel_mode, ParallelMode
        ), f"expected the argument parallel_mode to be of enum ParallelMode, but got {type(parallel_mode)}"

    def get_global_rank(self):
        """Returns the global rank of the current device.

        Returns:
            int: The global rank of the current device
        """
        return self._global_ranks[ParallelMode.GLOBAL]

    def get_local_rank(self, parallel_mode: ParallelMode):
        """Returns the local rank of the current device.

        Args:
            parallel_mode: The parallel mode for the rank.

        Returns:
            int: The local rank of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._local_ranks.get(parallel_mode, 0)

    def get_next_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the next device.

        Args:
            parallel_mode: The parallel mode for the rank.

        Returns:
            int: The global rank of the next device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the previous device.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The global rank of the previous device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_using_dp(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.DATA and its world_size is greater than 1.
        """
        return self.is_initialized(ParallelMode.DATA) and self.get_world_size(ParallelMode.DATA) > 1

    def is_using_tp(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.TENSOR and its world_size is greater than 1.
        """
        return self.is_initialized(ParallelMode.TENSOR) and self.get_world_size(ParallelMode.TENSOR) > 1

    def is_using_pp(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.PIPELINE and its world_size is greater than 1.
        """
        return self.is_initialized(ParallelMode.PIPELINE) and self.get_world_size(ParallelMode.PIPELINE) > 1

    def is_using_sequence(self):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.SEQUENCE and its world_size is greater than 1.
        """
        return False
        # return gpc.is_initialized(ParallelMode.SEQUENCE) and gpc.get_world_size(ParallelMode.SEQUENCE) > 1

    def is_first_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the first one
        among its group for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = 0
        if self.is_initialized(parallel_mode):
            rank = self.get_local_rank(parallel_mode)
        return rank == 0

    def is_rank_for_log(self):
        """Returns a boolean value indicating whether the current device should print log."""
        # is_log_rank = (
        #     self.is_first_rank(ParallelMode.DATA)
        #     and self.is_first_rank(ParallelMode.TENSOR)
        #     and self.is_last_rank(ParallelMode.PIPELINE)
        # )
        is_log_rank = (
            self.is_first_rank(ParallelMode.WEIGHT)
            and self.is_first_rank(ParallelMode.DATA)
            and self.is_first_rank(ParallelMode.WEIGHT_DATA)
        )
        return is_log_rank

    def is_last_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the last one
        among its group for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = 0
        world_size = 1
        if self.is_initialized(parallel_mode):
            rank = self.get_local_rank(parallel_mode)
            world_size = self.get_world_size(parallel_mode)
        return rank == world_size - 1

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size is not None and self.virtual_pipeline_parallel_rank != 0:
                return False
        return self.is_first_rank(ParallelMode.PIPELINE)

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if (
                self.virtual_pipeline_parallel_size is not None
                and self.virtual_pipeline_parallel_rank != self.virtual_pipeline_parallel_size - 1
            ):
                return False
        return self.is_last_rank(ParallelMode.PIPELINE)

    def is_no_pp_or_last_stage(self):
        # NOTICE!!!, this will ignore virutal stage
        return not self.is_initialized(ParallelMode.PIPELINE) or self.is_last_rank(ParallelMode.PIPELINE)

    def get_world_size(self, parallel_mode: ParallelMode):
        """Returns the world size for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The world size for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._world_sizes.get(parallel_mode, 1)

    def get_group(self, parallel_mode: ParallelMode):
        """Returns the group of the current device for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            torch.distributed.ProcessGroup: The group of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._groups[parallel_mode]

    def get_ranks_in_group(self, parallel_mode: ParallelMode):
        """Returns the rank of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The rank of the current device for `parallel_mode` in the group.
        """
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def get_cpu_group(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._cpu_groups[parallel_mode]

    def init_global_dist(self, rank: int, world_size: int, backend: str, host: str, port: int, use_cpu: bool = False):
        """Initializes the global distributed environment

        Args:
           rank (int): rank for the default process group.
           world_size (int): world size of the default process group.
           backend (str): backend for ``torch.distributed``
           host (str): the master address for distributed training.
           port (str): the master port for distributed training.
           use_cpu (bool): whether to set up cpu process group.
        """
        # None will give the default global process group for pytorch dist operations
        ranks = list(range(world_size))
        self._register_dist(rank, world_size, None, None, ranks, ParallelMode.GLOBAL)
        self._global_ranks[ParallelMode.GLOBAL] = rank

    def _register_dist(self, local_rank, world_size, process_group, cpu_group, ranks_in_group, mode):
        self._check_parallel_mode(mode)
        self._local_ranks[mode] = local_rank
        self._world_sizes[mode] = world_size
        self._groups[mode] = process_group
        self._cpu_groups[mode] = cpu_group
        self._ranks_in_group[mode] = ranks_in_group

    def check_sanity(self):
        """Checks sanity of the parallel context.

        Raises:
            AssertionError: Raises an AssertionError if the world size does not equal to the product
                of data parallel size, pipeline parallel size and tensor parallel size.
        """
        dps = self.data_parallel_size
        pps = self.pipeline_parallel_size
        tps = self.tensor_parallel_size
        ws = self.world_size
        # assert ws == dps * pps * tps, (
        #     f"Expected the world size {ws} to be equal to data"
        #     f" parallel size ({dps}) * pipeline parallel size "
        #     f"({pps}) * tensor parallel size ({tps})"
        # )
        assert self.zero1_parallel_size > 0, f"zero1_parallel_size: {self.zero1_parallel_size} should > 0"

        # check for fsdp:
        # if zo_size < dp_size, ckpts saving will introduce redundent storage for model weights
        # because pytorch "ShardTensor" need to ensure current global rank equals to saved shard's global rank
        # pytorch vision: 1.13.1+cu117
        if self.data_parallel_size > self.zero1_parallel_size and self.config.parallel.zero1.get("fsdp", False):
            print.warning(
                f"zo size: {self.zero1_parallel_size} < dp size: {self.data_parallel_size}, "
                "will introduce redundancy when saving fsdp model ckpts, recommend setting them to same value"
            )

    def _set_parallel_size_from_config(self, config: dict, key: str, attr_name: str):
        if key in config:
            ele = config[key]
            if isinstance(ele, int):
                setattr(self, attr_name, ele)
            elif isinstance(ele, dict):
                setattr(self, attr_name, ele["size"])
            else:
                raise NotImplementedError(
                    f'{"Parallel configuration does not support this kind of argument, please use int or dict"}'
                )

    def init_parallel_groups(self):
        """Initializes the parallel groups."""

        # get rank and world size
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)
        self.world_size = world_size

        # set parallel size as attributes for global context
        parallel_config = self.config.get("parallel", None)
        if parallel_config is not None:
            self._set_parallel_size_from_config(parallel_config, "weight", "weight_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "tensor", "tensor_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "pipeline", "pipeline_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "zero1", "zero1_parallel_size")

        # the user should not set the data parallel size manually
        # instead, it should be calculated based on other parallel config
        assert self.zero1_parallel_size >= 1, f"self.zero1_parallel_size: {self.zero1_parallel_size} should >= 1"
        self.sequence_parallel_size = self.tensor_parallel_size
        self.data_parallel_size = self.world_size // self.pipeline_parallel_size // self.sequence_parallel_size
        self.weight_data_parallel_size = self.world_size // self.pipeline_parallel_size // self.weight_parallel_size
        if parallel_config["tensor"]["mode"] != "isp":
            assert (
                self.zero1_parallel_size <= self.data_parallel_size
            ), f"zero1_size:{self.zero1_parallel_size} should be less than dp_size:{self.data_parallel_size}"
            assert (
                self.data_parallel_size % self.zero1_parallel_size == 0
            ), f"data_parallel_size:{self.data_parallel_size} % zero1_parallel_size: {self.zero1_parallel_size} != 0"
        else:
            assert (
                self.zero1_parallel_size <= self.weight_data_parallel_size
            ), f"zero1_size:{self.zero1_parallel_size} should be less than wdp_size:{self.weight_data_parallel_size}"
            assert (
                self.weight_data_parallel_size % self.zero1_parallel_size == 0
            ), f"weight_data_parallel_size:{self.weight_data_parallel_size} % zero1_parallel_size: {self.zero1_parallel_size} != 0"

        # the recommended nettest_parallel_size is 32 GPUs
        self.nettest_parallel_size = 32

        # assert (
        #     self.data_parallel_size % self.config.model.get("num_experts", 1) == 0
        #     or self.config.model.get("num_experts", 1) % self.data_parallel_size == 0
        # ), "can not place the experts evenly"

        # by default, expert_parallel_size equals to data_parallel_size, but if the number of experts is smaller
        # than data_parallel_size, set expert_parallel_size to be the number of experts to make sure each device
        # has one expert.
        self.expert_parallel_size = 1
        self.check_sanity()

        initializer_args = [
            rank,
            world_size,
            self.weight_parallel_size,
            self.weight_data_parallel_size,
            self.sequence_parallel_size,
            self.data_parallel_size,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.zero1_parallel_size,
            self.nettest_parallel_size,
            self.expert_parallel_size,
        ]

        # run initialization of different process groups
        initializers = []
        initializers.append(pgroup_initializer.Initializer_Weight(*initializer_args))
        if parallel_config["tensor"]["mode"] == "isp":
            initializers.append(pgroup_initializer.Initializer_Weight_Data(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Data(*initializer_args))
        # if self.weight_parallel_size <= 1:
        #     initializers.append(pgroup_initializer.Initializer_Model(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Tensor(*initializer_args))
        if parallel_config["tensor"]["mode"] != "isp":
            initializers.append(pgroup_initializer.Initializer_Zero1(*initializer_args))
        else:
            initializers.append(pgroup_initializer.Initializer_Zero1_ISP(*initializer_args))
        if isinstance(self.config.parallel.zero1, dict) and self.config.parallel.zero1.get("fsdp", False):
            initializers.append(pgroup_initializer.Initializer_Zero3_dp(*initializer_args))
        # initializers.append(pgroup_initializer.Initializer_Nettest(*initializer_args))
        if self.pipeline_parallel_size >= 1:
            initializers.append(pgroup_initializer.Initializer_Pipeline(*initializer_args))
        # if self.config.model.get("num_experts", 1) > 1:
        #     initializers.append(pgroup_initializer.Initializer_Expert_Data(*initializer_args))
        for initializer in initializers:
            parallel_setting = initializer.init_dist_group()
            if isinstance(parallel_setting, list):
                for args in parallel_setting:
                    self._register_dist(*args)
            else:
                self._register_dist(*parallel_setting)

    def is_initialized(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether `parallel_mode` is initialized
        in the current system.
        """
        return parallel_mode in self._groups

    def destroy(self):
        self._groups.clear()

    def set_device(self, device_ordinal: int = None):
        """Sets distributed processes to be bound to devices.

        Args:
           device_ordinal (int, optional): the device id to be bound to
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node

        torch.cuda.set_device(device_ordinal)
        print.info(f"process rank {global_rank} is bound to host:{socket.gethostname()} device: {device_ordinal}")

    def set_virtual_pipeline_parallel_size(self, size):
        self.virtual_pipeline_parallel_size = size

    def set_virtual_pipeline_parallel_rank(self, rank):
        self.virtual_pipeline_parallel_rank = rank

    def check_pg_is_intra(self, parallel_mode: ParallelMode):
        pg_group_ranks = self.get_ranks_in_group(parallel_mode)
        if len(pg_group_ranks) > 8:
            return False
        else:
            min_rank = min(pg_group_ranks)
            max_rank = max(pg_group_ranks)
            return (max_rank - min_rank) <= 7

    def same_group_in_one_node(self, parallel_mode: ParallelMode):
        """获得一个节点内有多少个相同类型的PG, 在跨节点通信时会存在带宽竞争
        这里返回的相同PG的数量会乘上每个rank的通信数据量大小

        Args:
            parallel_mode (ParallelMode):

        Returns:
            int: 一个节点内相同类型的PG的数量
        """
        pg_group_ranks = self.get_ranks_in_group(parallel_mode)
        pg_group_ranks = sorted(pg_group_ranks)
        if len(pg_group_ranks) == 1:
            return 1
        else:
            stride = pg_group_ranks[1] - pg_group_ranks[0]
            if stride >= 8:
                return 8
            else:
                return stride


global_context = ParallelContext()
