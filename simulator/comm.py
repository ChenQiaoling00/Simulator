from simulator.ab_cost_model import (
    allgather,
    allreduce,
    alltoall,
    get_comm_cost,
    reducescatter,
)
import sys
sys.path.append("..")
from simulator.context import ParallelMode
from simulator.context import global_context as gpc
from utils.common import AlgoType, CostType
from utils.utils import CommPredict


class TransformerCommunication:
    def __init__(
        self,
        b,
        s,
        h,
        vocab_size,
        mlp_ratio,
        multiple_of,
        dtype_size,
        ckpt=0,
        wdp_size=1,
    ):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size

        self.qkv_communication_latency = 0
        self.post_attention_communication_latency = 0
        self.first_linear_communication_latency = 0
        self.second_linear_communication_latency = 0
        self.attention_all_to_all_communication_latency = 0

        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.dtype_size = dtype_size
        self.mlp_hidden_size = self.multiple_of * (
            (int(self.h * self.mlp_ratio) + self.multiple_of - 1) // self.multiple_of
        )

        self.ckpt = ckpt  # activation checkpoint

        # self.toal_comm = self.communication_isp()

    def communication_isp(self):
        """
        ckpt: means the activation checkpoint, {0 or 1}

        sp communication:

        comm(sp) = comm(forward, sp) + comm(backward, sp)

        comm(forward, sp) = 4 * comm(all2all, s/sp, b, h) * (ckpt + 1)

        comm(backward, sp) = 4 * comm(all2all, s/sp, b, h)

        wp communication: (In our implementation, the wp communication of ckpt==1 is the same as ckpt==0)

        comm(wp) = comm(forwad, wp) + comm(backward, wp)

        comm(forward, wp) = comm(all_gather, (wqkv, wo, mlp))

        comm(backward, wp) = comm(all_gather, (wqkv, wo, mlp)) + comm(reduceScatter, (wqkv, wo, mlp))

        wdp communication: (actually wdp communication should be included in the optimizer communication)
        """

        self.wp_scale = gpc.get_world_size(ParallelMode.WEIGHT)
        self.sp_scale = gpc.get_world_size(ParallelMode.TENSOR)

        # wp communication
        qkv_wp_volume = 3 * self.dtype_size * self.h**2
        wo_wp_volume = self.dtype_size * self.h**2
        mlp_w1_volume = self.dtype_size * self.h * self.mlp_hidden_size

        qkv_latency = 2 * allgather(qkv_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            qkv_wp_volume, ParallelMode.WEIGHT
        )
        wo_latency = 2 * allgather(wo_wp_volume, ParallelMode.WEIGHT) + reducescatter(wo_wp_volume, ParallelMode.WEIGHT)
        mlp_w1_latency = 2 * allgather(mlp_w1_volume, ParallelMode.WEIGHT) + reducescatter(
            mlp_w1_volume, ParallelMode.WEIGHT
        )
        mlp_w2_latency = mlp_w1_latency
        mlp_w3_latency = mlp_w1_latency

        # sp communication
        all2all_volume = self.s / self.sp_scale * self.b * self.h * self.dtype_size
        all2all_latency = alltoall(all2all_volume, ParallelMode.TENSOR)

        wp_comm_latency = qkv_latency + wo_latency + mlp_w1_latency + mlp_w2_latency + mlp_w3_latency
        sp_comm_latency = 4 * all2all_latency * (self.ckpt + 1) + 4 * all2all_latency  # forward + backward

        # wdp communication
        # wdp_volume = self.model_para / gpc.get_world_size(ParallelMode.WEIGHT_DATA)  # TODO: 这个通信量是否合理?
        # wdp_latency = allreduce(wdp_volume, ParallelMode.WEIGHT_DATA)

        return wp_comm_latency, sp_comm_latency

    def communication_msp(self):
        """
        ckpt: means the activation checkpoint, {0 or 1}

        sp communication:

        comm(sp) = comm(forward, sp) + comm(backward, sp)

        comm(forward, sp) = (2 * comm(all_gather, s, b, h) + 2 * comm(reduceScatter, s, b, h)) * (ckpt + 1)

        comm(backward, sp) = 2 * comm(reduceScatter, s, b, h) + 2 * comm(all_gather, s, b, h)

        wp communication:

        comm(wp) = comm(forwad, wp) + comm(backward, wp)

        comm(forward, wp) = comm(all_gather, (wqkv, wo, mlp))

        comm(backward, wp) = comm(all_gather, (wqkv, wo, mlp)) + comm(reduceScatter, (wqkv, wo, mlp))

        wdp communication: (actually wdp communication should be included in the optimizer communication)
        """
        self.wp_scale = gpc.get_world_size(ParallelMode.WEIGHT)
        self.sp_scale = gpc.get_world_size(ParallelMode.TENSOR)

        # compute sp communication
        # all_gather and reduceScatter have the same commu volume
        # the communication volume in backward is equal to the forward
        qkv_sp_volume = self.s * self.b * self.h * self.dtype_size  # the forward all-gather
        wo_sp_volume = self.s * self.b * self.h * self.dtype_size  # the forward reduceScatter
        mlp_w1_sp_volume = qkv_sp_volume  # the forward all-gather
        mlp_w2_sp_volume = self.s * self.b * self.h * self.dtype_size  # the forward reduceScatter

        # compute the sp latency (forward + backward)
        sp_forward = (
            allgather(qkv_sp_volume, ParallelMode.TENSOR)
            + reducescatter(wo_sp_volume, ParallelMode.TENSOR)
            + allgather(mlp_w1_sp_volume, ParallelMode.TENSOR)
            + reducescatter(mlp_w2_sp_volume, ParallelMode.TENSOR)
        )

        sp_backward = (
            reducescatter(qkv_sp_volume, ParallelMode.TENSOR)
            + allgather(wo_sp_volume, ParallelMode.TENSOR)
            + reducescatter(mlp_w1_sp_volume, ParallelMode.TENSOR)
            + allgather(mlp_w2_sp_volume, ParallelMode.TENSOR)
        )

        sp_forward = sp_forward * (self.ckpt + 1)

        sp_comm_latency = sp_forward + sp_backward

        # commpute wp communication
        qkv_wp_volume = 3 * self.h * self.h / self.sp_scale * self.dtype_size
        wo_wp_volume = self.h * self.h / self.sp_scale * self.dtype_size

        # w2 and w3 have the same volume as w1
        mlp_w1_wp_volume = self.h * self.mlp_hidden_size / self.sp_scale * self.dtype_size

        qkv_wp_latency = 2 * allgather(qkv_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            qkv_wp_volume, ParallelMode.WEIGHT
        )
        wo_wp_latency = 2 * allgather(wo_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            wo_wp_volume, ParallelMode.WEIGHT
        )
        mlp_w1_wp_latency = 2 * allgather(mlp_w1_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            mlp_w1_wp_volume, ParallelMode.WEIGHT
        )
        mlp_w2_wp_latency = mlp_w1_wp_latency

        wp_comm_latency = qkv_wp_latency + wo_wp_latency + mlp_w1_wp_latency + mlp_w2_wp_latency

        # wdp communication
        # wdp_volume = self.model_para // self.sp_scale // self.wp_scale  # TODO: 这个通信量是否合理?
        # wdp_latency = allreduce(wdp_volume, ParallelMode.WEIGHT_DATA)

        return wp_comm_latency, sp_comm_latency

    def communication_fsp(self):
        """
        ckpt: means the activation checkpoint, {0 or 1}

        sp communication:

        comm(sp) = comm(forward, sp) + comm(backward, sp)

        comm(forward, sp) = (2 * comm(all_gather, s, b, h) + 2 * comm(reduceScatter, s, b, h)) * (ckpt + 1)

        comm(backward, sp) = 2 * comm(reduceScatter, s, b, h) + 4 * comm(all_gather, s, b, h)

        wp communication:

        comm(wp) = comm(forwad, wp) + comm(backward, wp)

        comm(forward, wp) = comm(all_gather, (wqkv, wo, mlp))

        comm(backward, wp) = comm(all_gather, (wqkv, wo, mlp)) + comm(reduceScatter, (wqkv, wo, mlp))

        wdp communication: (actually wdp communication should be included in the optimizer communication)
        """

        self.wp_scale = gpc.get_world_size(ParallelMode.WEIGHT)
        self.sp_scale = gpc.get_world_size(ParallelMode.TENSOR)

        # compute sp communication
        # all_gather and reduceScatter have the same commu volume
        # the communication volume in backward is equal to the forward
        qkv_sp_volume = self.s * self.b * self.h * self.dtype_size  # the forward all-gather
        wo_sp_volume = self.s * self.b * self.h * self.dtype_size  # the forward reduceScatter
        mlp_w1_sp_volume = qkv_sp_volume  # the forward all-gather
        mlp_w2_sp_volume = self.s * self.b * self.h * self.dtype_size  # the forward reduceScatter

        # compute the sp latency (forward + backward)
        sp_forward = (
            allgather(qkv_sp_volume, ParallelMode.TENSOR)
            + reducescatter(wo_sp_volume, ParallelMode.TENSOR)
            + allgather(mlp_w1_sp_volume, ParallelMode.TENSOR)
            + reducescatter(mlp_w2_sp_volume, ParallelMode.TENSOR)
        )

        sp_backward = (
            allgather(qkv_sp_volume, ParallelMode.TENSOR)
            + reducescatter(qkv_sp_volume, ParallelMode.TENSOR)
            + allgather(wo_sp_volume, ParallelMode.TENSOR)
            + allgather(mlp_w1_sp_volume, ParallelMode.TENSOR)
            + reducescatter(mlp_w1_sp_volume, ParallelMode.TENSOR)
            + allgather(mlp_w2_sp_volume, ParallelMode.TENSOR)
        )

        sp_forward = sp_forward * (self.ckpt + 1)

        sp_comm_latency = sp_forward + sp_backward

        # commpute wp communication
        qkv_wp_volume = 3 * self.h * self.h / self.sp_scale * self.dtype_size
        wo_wp_volume = self.h * self.h / self.sp_scale * self.dtype_size

        # w2 and w3 have the same volume as w1
        mlp_w1_wp_volume = self.h * self.mlp_hidden_size / self.sp_scale * self.dtype_size

        qkv_wp_latency = 2 * allgather(qkv_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            qkv_wp_volume, ParallelMode.WEIGHT
        )
        wo_wp_latency = 2 * allgather(wo_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            wo_wp_volume, ParallelMode.WEIGHT
        )
        mlp_w1_wp_latency = 2 * allgather(mlp_w1_wp_volume, ParallelMode.WEIGHT) + reducescatter(
            mlp_w1_wp_volume, ParallelMode.WEIGHT
        )
        mlp_w2_wp_latency = mlp_w1_wp_latency

        wp_comm_latency = qkv_wp_latency + wo_wp_latency + mlp_w1_wp_latency + mlp_w2_wp_latency

        # wdp communication
        # wdp_volume = self.model_para // self.sp_scale // self.wp_scale  # TODO: 这个通信量是否合理?
        # wdp_latency = allreduce(wdp_volume, ParallelMode.WEIGHT_DATA)

        return wp_comm_latency, sp_comm_latency

    def communication(self, algo_type):
        if algo_type == AlgoType.ISP:
            return self.communication_isp()
        elif algo_type == AlgoType.MSP:
            return self.communication_msp()
        elif algo_type == AlgoType.FSP:
            return self.communication_fsp()
        raise ValueError(f"Unkown algo_type: {algo_type}")
