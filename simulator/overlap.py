from simulator.comm import TransformerCommunication
from simulator.comp import TransformerComputation
from utils.common import get_model_config


class TransformerOverlapOneLayer:
    def __init__(
        self,
        micro_bsz,
        seq_len,
        vocab_size,
        dtype_size,
        sp_size,
        pp_size,
        world_size,
        ckpt,
        hidden_dim,
        num_head,
        mlp_ratio,
        multiple_of,
    ):
        self.b = micro_bsz  # Batch size
        self.s = seq_len  # Sequence length
        self.vocab_size = vocab_size
        self.sp_scale = sp_size
        self.dtype_size = dtype_size
        self.world_size = world_size
        self.pp_size = pp_size

        self.h, self.a, self.mlp_ratio, self.multiple_of = hidden_dim, num_head, mlp_ratio, multiple_of

        self.ckpt = ckpt  # the activation checkpoint

    def _get_overlap(self, algo_type):
        # Communication latency of a transformer layer (forward + backward)
        comm_wp, comm_sp = TransformerCommunication(
            self.b,
            self.s,
            self.h,
            self.vocab_size,
            dtype_size=self.dtype_size,
            mlp_ratio=self.mlp_ratio,
            multiple_of=self.multiple_of,
            ckpt=self.ckpt,
        ).communication(algo_type)

        # Computation latency of a transformer layer (forward + backward)
        comp_wp, comp_attn = TransformerComputation(
            self.a,
            self.b,
            self.s,
            self.h,
            self.vocab_size,
            dtype_size=self.dtype_size,
            mlp_ratio=self.mlp_ratio,
            multiple_of=self.multiple_of,
            sp_scale=self.sp_scale,
            ckpt=self.ckpt,
        ).total_computation(algo_type)

        return comm_wp, comm_sp, comp_wp, comp_attn
