"""
main package
"""
import pickle

# from simulator.simulator import Constraint, Simulator
from simulator.noz3_simulator import Constraint
from utils.config import Config


def main():
    """main function"""
    GPU_NUMS = 128
    MIN_GLOBA_BSZ = 4*1024*1024
    MAX_GLOBA_BSZ = 4*1024*1024
    config = Config(
        {
            "world_size_max": GPU_NUMS,
            "world_size_min": GPU_NUMS,
            "global_bsz": 1 * 1024**32,
            "global_bsz_min": MIN_GLOBA_BSZ,
            "global_bsz_max": MAX_GLOBA_BSZ,
            "sequence_length": 128* 1024,
            "model_size": 13,
            "vocab_size": 50256,
            "dtype_size": 2,
            "use_fa": 1,
            "fixed_micro_num": 1,
            "fixed_micro_bsz": 1,
            "mem_threshold": 8000 * 1024**3,
            "wp_penalty_coefficient": 0.2,
        }
    )

    # global_bsz (int): Global batch size, used when use_strict_bsz is True.
    # global_bsz_min (int): global_bsz's upper bound on the number of searches.
    # global_bsz_max (int): global_bsz's lower bound for searching
    # max_world_size (int): upper bound of world_size for the search
    # min_world_size (int): lower bound for world_size searches
    # seq_len (int).
    # overlap_wdp (int): whether or not to consider overlap wdp communication
    # fixed_micro_num (int): if or not to fix micro_num, default is None.
    # fixed_micro_bsz (int): if or not to fix micro_bsz, default is None.
    # use_strict_bsz (bool): if True, will strictly limit globa bsz to the value of the global_bsz parameter
    # debug (bool): if or not to output additional debug information
    # config (dict): the config of the model

    externl_sim = Constraint(
        debug=True,
        overlap_wdp=True,
        use_fixed_micro_bsz=False,
        use_strict_bsz=False,
        config=config,
    )
    externl_sim.run_flexible_worldsize_loop()


if __name__ == "__main__":
    main()
