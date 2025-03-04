from simulator.predict_cost_model import GenCostModel
from utils.common import (CostType, build_process_gourp, get_global_rank,
                          get_world_size)
from utils.config import Config

if __name__ == "__main__":
    world_size = get_world_size()
    if world_size > 1:
        build_process_gourp(world_size)

    build_type_list = [CostType.FLASH_ATTN] # CostType.LINEAR, CostType.ALL2ALL, CostType.ALLREDUCE, CostType.REDUCESCATTER, CostType.ALLGATHER, CostType.BROADCAST
    # build_type_list = [CostType.ALLGATHER]
    cost_model = GenCostModel(
        is_master=get_global_rank() == 0, build_type_list=build_type_list
    )
    cost_model.build_cost_model_by_key_value()
    cost_model.dump_data()
