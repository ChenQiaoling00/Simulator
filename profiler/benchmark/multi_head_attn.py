#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch
from einops import rearrange
from torch import nn

from profiler.registry import BENCHMARK_INITIALIZER
from utils.common import TP_SIZE_RANGE, K, get_local_rank

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    from flash_attn.modules.mha import FlashSelfAttention, SelfAttention
except ModuleNotFoundError:
    flash_attn_qkvpacked_func = None
    FlashSelfAttention = None
    SelfAttention = None
    print("import fa failed!", flush=True)

from .base_benchmark import UnitBench

BENCH_TYPE = "flash_attn"


class MHA(nn.Module):
    def __init__(self, head_dim, causal=True, attn_drop_rate=0) -> None:
        super().__init__()
        self.softmax_scale = 1 / math.sqrt(head_dim)
        self.causal = causal
        self.head_dim = head_dim
        self.inner_attn = FlashSelfAttention(causal=causal)

    def forward(self, qkv, cu_seqlens, max_seqlen):
        qkv = rearrange(qkv, "t (three h d) -> t three h d", three=3, d=self.head_dim)
        context = self.inner_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, causal=self.causal)
        context = rearrange(context, "b h d -> b (h d)")
        return context


@BENCHMARK_INITIALIZER.register_module(module_name=BENCH_TYPE)
class UnitMultiHeadAttn(UnitBench):
    test_loop = {
        "seq_len": [256 * K, int(0.25 * K), int(0.5 * K), 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K, 128 * K],
        "num_heads_and_hidden_dim": [(80, 10240), (64, 8192), (48, 6144), (40, 5120), (32, 4096)],  #
        # "num_heads_and_hidden_dim": [(80, 10240)],  # , 256 * K
        "dtype": [torch.bfloat16],
        "micro_bsz": [1, 2, 4, 8],
        "tp_size": TP_SIZE_RANGE,
        "is_fwd": [True, False],
    }

    # test_loop = {
    #     "seq_len": [256 * K],
    #     "num_heads_and_hidden_dim": [(64, 8192)],  #
    #     # "num_heads_and_hidden_dim": [(80, 10240)],  # , 256 * K
    #     "dtype": [torch.bfloat16],
    #     "micro_bsz": [1],
    #     "tp_size": [16],
    #     "is_fwd": [True, False],  # fwd: 340ms, bwd: 870ms
    # }

    def __init__(self, seq_len, num_heads_and_hidden_dim, dtype, micro_bsz, tp_size, is_fwd) -> None:
        num_heads, embed_dim = num_heads_and_hidden_dim
        self.num_heads_and_hidden_dim = num_heads_and_hidden_dim
        self.tp_size = tp_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.dtype_size = 2 if self.dtype == torch.bfloat16 else 4
        self.micro_bsz = micro_bsz
        self.oom = False
        self.is_fwd = is_fwd

        assert self.embed_dim % self.tp_size == 0
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % self.tp_size == 0, "num_heads must be divisible by tp_size"
        assert num_heads >= tp_size, f"head nums must bigger then tp_size: {tp_size}"

        self.num_atten_head_tp = num_heads // self.tp_size
        self.head_dim = self.embed_dim // num_heads
        self.tp_embedding_dim = self.embed_dim // self.tp_size

        self.packed_length = self.seq_len * self.micro_bsz
        self.device = f"cuda:{get_local_rank()}"

        indexs, cu_seqlens = [], [0]
        cu_seqlens = [i * self.seq_len for i in range(self.micro_bsz + 1)]
        indexs = list(range(self.seq_len)) * self.micro_bsz

        weights_mem_used = self.packed_length * 3 * self.embed_dim * self.dtype_size
        attn_activation = 11 * self.packed_length * self.embed_dim
        mem_used = attn_activation + weights_mem_used

        oom = False
        if mem_used > 75 * 1024**3:
            oom = True

        # 约束1: seqlen最大不能超过256K(不含)
        # 约束2: embed_dim在被tp切过之后若大于6144， 则packed_length不能大于256k
        if self.packed_length >= 256 * K and (self.embed_dim / self.tp_size) >= 6144:
            oom = True
        if self.seq_len >= 256 * K and self.micro_bsz > 1:
            oom = True
        if self.packed_length >= 524288 and (self.embed_dim / self.tp_size) >= 3072:
            oom = True
        if self.packed_length >= 1048576 and (self.embed_dim / self.tp_size) >= 2048:
            oom = True

        if oom:
            assert (
                False
            ), f"warning : mem_used: {mem_used/1024**3:.2f} GB, seq_len: {self.seq_len}, embed_dim: {self.embed_dim}, tp_size: {self.tp_size}"

        assert self.tp_embedding_dim == self.num_atten_head_tp * self.head_dim

        self.qkv = torch.rand(
            size=(self.packed_length, 3 * self.tp_embedding_dim),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        self.dtype_size = self.qkv.element_size()
        self.indexs = torch.tensor(data=indexs, dtype=torch.int32, device=self.device)
        self.cu_seqlens = torch.tensor(data=cu_seqlens, dtype=torch.int32, device=self.device)
        self.MHA = MHA(head_dim=self.head_dim, causal=True)
        if not self.is_fwd:
            self.output = self.run_fwd()
            self.grad = torch.randn_like(self.output) / 32  # avoid grad is too large.

    def run(self):
        if self.is_fwd:
            self.run_fwd()
        else:
            self.run_bwd(self.output, self.grad)

    def run_fwd(self):
        context = self.MHA(self.qkv, cu_seqlens=self.cu_seqlens, max_seqlen=self.seq_len)
        return context

    def run_bwd(self, output, grad):
        output.backward(grad, retain_graph=True)

    @staticmethod
    def gen_store_key(micro_bsz, seq_len, num_heads_and_hidden_dim, tp_size, is_fwd):
        _, embed_dim = num_heads_and_hidden_dim
        tp_embedding_dim = embed_dim // tp_size
        return f"b_{micro_bsz}_s_{seq_len}_h_{tp_embedding_dim}_fwd_{is_fwd}"

    def complexity(self):
        return UnitMultiHeadAttn.gen_store_key(
            self.micro_bsz, self.seq_len, self.num_heads_and_hidden_dim, self.tp_size, self.is_fwd
        )
        # return f"{self.seq_len} * {self.hidden_dim} * {self.hidden_dim}"
