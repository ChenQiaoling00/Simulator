import functools

from utils.common import BW, CostType

import sys
sys.path.append("..")
from simulator.context import ParallelMode
from simulator.context import global_context as gpc
from simulator.predict_cost_model import SplineModel

cost_model = None
# cost_model = None
scale_ratio = [1.415134488, 1.208864145, 1.1, 1]


def coll_algo_bw(comm_op, size, n):
    if comm_op == CostType.ALL2ALL:
        if n <= 8:
            return size * (n - 1) / n
        else:
            # intra_parts = 8
            one_part = size / n
            return 8 * one_part * (n - 8 / n)
    elif comm_op == CostType.ALLREDUCE:
        return size * 2 * (n - 1) / n
    elif comm_op == CostType.REDUCESCATTER:
        return size * (n - 1) / n
    elif comm_op == CostType.ALLGATHER:
        return size * (n - 1) / n
    elif comm_op == CostType.BROADCAST:
        return size * (n - 1) / n
    elif comm_op == CostType.P2P:
        return size

    raise ValueError(f"unknown comm_op: {comm_op}")


def coll_bus_bw(comm_op, size, n):
    if comm_op == CostType.ALL2ALL:
        return size
    elif comm_op == CostType.ALLREDUCE:
        return size * 2
    elif comm_op == CostType.REDUCESCATTER:
        return size
    elif comm_op == CostType.ALLGATHER:
        return size
    elif comm_op == CostType.BROADCAST:
        return size
    elif comm_op == CostType.P2P:
        return size

    raise ValueError(f"unknown comm_op: {comm_op}")


# 需要判断是否打满带宽
def get_scale_ratio(scale):
    # 通信扩展惩罚系数
    if scale <= 16:
        return 1
    elif 16 < scale <= 32:
        return 1.1
    elif 32 < scale <= 64:
        return 1.2
    elif 64 < scale <= 256:
        return 1.3
    elif 256 < scale <= 512:
        return 1.4
    else:
        return 1.5


def get_comm_cost_logic(comm_volume: int, parallel_mode: ParallelMode, comm_op: CostType = None):
    """根据通信量获得近似的通信延迟,这个函数考虑了跨节点带宽content的情景
    所以为了正确计算延迟，传入的 comm_volume 必须是以单个rank视角下的通信量
    (即代码中实际传入的通信量)

    Args:
        comm_volume (int): 通信量, 单位B
        parallel_mode (ParallelMode): gpc并行模式
        comm_op (CostType, optional): 通信算子

    Returns:
        int: 通信延迟,是乘以10**4后并取整后的数值
    """
    scale = gpc.get_world_size(parallel_mode)

    if parallel_mode == ParallelMode.PIPELINE:
        scale = 2

    if scale <= 1:
        return 0

    is_intra = gpc.check_pg_is_intra(parallel_mode)
    if not is_intra:
        num_partner = gpc.same_group_in_one_node(parallel_mode)
        assert num_partner <= 8, f"num_partner: {num_partner}"
        if parallel_mode == ParallelMode.WEIGHT:
            assert num_partner == 1
        if parallel_mode == ParallelMode.TENSOR:
            assert num_partner == 1
        comm_volume *= num_partner

    global cost_model
    if cost_model is None:
        cost_model = SplineModel()

    # if comm_op == CostType.P2P:
    bw = BW.A800_NVL if is_intra else (BW.IB / get_scale_ratio(scale))
    return coll_algo_bw(comm_op, comm_volume, scale) / bw  # 转换成ms小数点保留两位
    # else:
    #     latency = cost_model.predict_cost(cost_type=comm_op, complexity=comm_volume, world_size=scale)
    #     print(f"comm_op: {comm_op}, world_size:{scale}, comm_volume: {comm_volume/1024**2:.3f} MB, latency: {latency*1000:.2f} ms")
    #     return latency


def get_predict_or_kv_cost(cost_type: CostType, complexity=0, **kwargs):
    global cost_model
    if cost_model is None:
        cost_model = SplineModel()

    return cost_model.predict_cost(cost_type, complexity=complexity, **kwargs)


get_comm_cost = get_comm_cost_logic

allgather = functools.partial(get_comm_cost, comm_op=CostType.ALLGATHER)
reducescatter = functools.partial(get_comm_cost, comm_op=CostType.REDUCESCATTER)
broadcast = functools.partial(get_comm_cost, comm_op=CostType.BROADCAST)
p2p = functools.partial(get_comm_cost, comm_op=CostType.P2P)
alltoall = functools.partial(get_comm_cost, comm_op=CostType.ALL2ALL)
allreduce = functools.partial(get_comm_cost, comm_op=CostType.ALLREDUCE)
