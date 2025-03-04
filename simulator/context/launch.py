import sys
sys.path.append("..")

from simulator.context import global_context as gpc
from simulator.context import ParallelMode

def check_and_modify_parallel_config(config_dict):
    # process the parallel config
    if "sequence_parallel" not in gpc.config.parallel:
        gpc.config.parallel._add_item("sequence_parallel", False)
    else:
        assert not (
            gpc.config.parallel.sequence_parallel is True and gpc.config.model.use_flash_attn is False
        ), "sequence parallel does not support use_flash_attn=False"

    # set default value for tensor parallel
    if isinstance(gpc.config.parallel["tensor"], int):
        gpc.config.parallel["tensor"] = dict(size=gpc.config.parallel["tensor"], mode="mtp")
    if gpc.config.parallel["tensor"].get("mode", None) is None:
        gpc.config.parallel["tensor"]["mode"] = "mtp"
    assert gpc.config.parallel["tensor"].get("mode", None) in [
        "mtp",
        "msp",
        "fsp",
        "isp",
    ], "invalid tensor parallel mode, only ['mtp', 'msp', 'fsp', 'isp'] is supported"

    # adapt to old version's sequence parallel config
    if gpc.config.parallel["tensor"].get("mode", None) in ["msp", "fsp", "isp"]:
        gpc.config.parallel.sequence_parallel = True

    # set default value for weight parallel
    if gpc.config.parallel["weight"].get("overlap", None) is None:
        gpc.config.parallel["weight"]["overlap"] = False
    if gpc.config.parallel["weight"].get("memory_pool", None) is None:
        gpc.config.parallel["weight"]["memory_pool"] = False
    if gpc.config.parallel["tensor"]["mode"] != "isp":
        assert gpc.config.parallel["weight"]["size"] <= 1, "weight parallel is only supported with isp"


    # # currently only interleaved pipeline scheduler with overlap can guarantee loss accuracy
    # if hasattr(gpc.config.model, "num_chunks") and gpc.config.model.num_chunks > 1:
    #     assert (
    #         gpc.config.parallel["pipeline"].get("interleaved_overlap", False) is True
    #     ), "only support interleaved pipeline scheduler with overlap"


    # 313->336



