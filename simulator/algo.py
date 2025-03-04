from simulator.overlap import TransformerOverlap
from utils.common import AlgoType, CostType


class BaseAlgo:
    def __init__(
        self,
        config: dict,
        cost_data: object,
        model_config: dict,
        X: list = [],
        C: list = [],
        A: list = [],
        num_strategies: int = 0,
    ) -> None:

        self._world_size = config["world_size"]
        self._global_batch_size = config["global_batch_size"]
        self._sequence_length = config["sequence_length"]
        self._model_size = config["model_size"]
        self._grad_acc = config["grad_acc"]
        self._SP = config["SP"]
        self._micro_batch_size = config["micro_bs"]
        self._vocab_size = config["vocab_size"]
        self._dtype_size = 2  # the sizeof(model.dtype)
        self._os_size_ratio = 2 if self._dtype_size == 2 else 1  # the sizeof(OS.P)
        self._p_size = self._model_size * 10**9  # model size
        self._l = model_config["l"]
        self._h = model_config["h"]
        self._a = model_config["a"]
        self._mlp_ratio = model_config["mlp_ratio"]
        self._multiple_of = model_config["multiple_of"]
        self._cost_data = cost_data
        self._num_strategies = num_strategies

        self.overlap_res = TransformerOverlap(
            b=self._dtype_size,
            s=self._sequence_length,
            h=self._h,
            # a=self._a,
            num_layers=self._l,
            dtype_size=self._dtype_size,
            mlp_ratio=self._mlp_ratio,
            multiple_of=self._multiple_of,
            vocab_size=self._vocab_size,
            cost_data=self._cost_data,
        )

        # the combination of parallel strategies
        # X[i][j]: i->P,G,OS; j->2,4,6,...
        self.X = X
        # the communication cost
        # C[i][j] means the communication cost of stratege X[i][j]
        self.C = C
        # the memory cost
        # A[i][j] means the memory cost of stratege X[i][j]
        self.A = A

    def _lookup_comm_cost(self, type: CostType, world_size, complexity):
        return self._cost_data[type].predict(world_size, complexity)

    def get_XCA(self):
        return self.X, self.C, self.A

    def set_memory_threshold(self):
        """set the memory threshold"""
        pass

    def get_comm_cost(self):
        """get the communication cost"""
        pass

    def get_mem_cost(self):
        """get the memory cost"""
        pass


class ISP(BaseAlgo):
    def __init__(
        self, config: dict, cost_data: object, model_config: dict, X: dict, C: dict, A: dict, num_strategies: int
    ) -> None:
        super().__init__(config, cost_data, model_config, X, C, A, num_strategies)
        self.algo_type = AlgoType.ISP

    def set_memory_threshold(self):
        self._activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (34 + (5 * self._a * self._sequence_length / self._h))
            / self._SP
        )
        self._memory_threshold = 80 * (1024**3) - self._activation
        if self._memory_threshold < 0:
            print(f"!!!warning!!!: self._memory_threshold: {self._memory_threshold} < 0")
        print(f"activation: {self._activation:.4f} GB")
        return self._memory_threshold

    def _get_os_comm_cost(self, comm_range):
        if comm_range <= 1:
            return 0
        comm_cost = self._dtype_size * self._p_size
        return self._lookup_comm_cost(CostType.ALLGATHER, comm_range, comm_cost)  # TODO: Should be broadcast

    def _comm_cost(self, i: int, j: int):
        """
        Get communication cost.

        Args:
            i (int): p (i==0), g (i==1), os (i==2)
            j (int): node count

        Returns:
            float: communication cost

        commu cost = fwd + bwd + optimizer

        fwd = sp + wp
        bwd = sp + wp
        optimizer = zp

        其中 wp的通信可以overlap
        """
        # self._SP_comm = self._get_sp_comm_cost(self._SP)

        if j != 0:
            comm_range = j * 8
        else:
            comm_range = 1  # no comm cost

        # 算overlap的通信开销
        overlap_cost = self.overlap_res._get_overlap(comm_range, self._SP, self.algo_type)

        # 算os的通信开销
        if comm_range == 1:
            os_comm_cost = 0
        else:
            os_comm_cost = self._get_os_comm_cost(comm_range)

        # 总的通信开销
        comm_cost = os_comm_cost + overlap_cost

        return comm_cost

    def get_comm_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                # TODO：这里需要支持更多的切分策略
                if j != 1 and j % 2 != 0:  # 节点数为奇数的时候
                    self.C[i][j] = self.C[i][j - 1] * 1.2
                else:  # 节点数为偶数
                    self.C[i][j] = self._comm_cost(i, j)

    def _mem_cost(self, i: int, j: int):
        if i == 0:
            if j == 0:
                # 不切P
                return self._dtype_size * self._model_size
            # 对P切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        elif i == 1:
            if j == 0:
                # 不切G
                return self._dtype_size * self._model_size
            # 对G切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        else:
            if j == 0:
                # 不切OS
                return self._dtype_size * self._os_size_ratio * 3 * self._model_size
            # 对OS切j*8份
            return self._dtype_size * self._os_size_ratio * 3 * self._model_size / (j * 8)

    def get_mem_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                if j != 1 and j % 2 != 0:
                    self.A[i][j] = self.A[i][j - 1] * 0.8
                else:
                    self.A[i][j] = self._mem_cost(i, j)


class MSP(BaseAlgo):
    def __init__(
        self, config: dict, cost_data: object, model_config: dict, X: dict, C: dict, A: dict, num_strategies: int
    ) -> None:
        super().__init__(config, cost_data, model_config, X, C, A, num_strategies)
        self.algo_type = AlgoType.MSP

    def set_memory_threshold(self):
        self._activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (4 + 30 / self._SP + (5 * self._a * self._sequence_length / self._h / self._SP))
        )
        self._memory_threshold = 80 * (1024**3) - self._activation
        if self._memory_threshold < 0:
            print(f"!!!warning!!!: self._memory_threshold: {self._memory_threshold} < 0")
        print(f"activation: {self._activation:.4f} GB")
        return self._memory_threshold

    def _get_os_comm_cost(self, comm_range):
        if comm_range <= 1:
            return 0
        comm_cost = self._dtype_size * self._p_size
        return self._lookup_comm_cost(CostType.ALLGATHER, comm_range, comm_cost)  # TODO: Should be broadcast

    def _comm_cost(self, i: int, j: int):
        """
        Get communication cost.

        Args:
            i (int): p (i==0), g (i==1), os (i==2)
            j (int): node count

        Returns:
            float: communication cost

        commu cost = fwd + bwd + optimizer

        fwd = sp + wp
        bwd = sp + wp
        optimizer = zp

        其中 wp的通信可以overlap
        """
        # self._SP_comm = self._get_sp_comm_cost(self._SP)

        if j != 0:
            comm_range = j * 8
        else:
            comm_range = 1  # no comm cost

        # 算overlap的通信开销
        overlap_cost = self.overlap_res._get_overlap(comm_range, self._SP, self.algo_type)

        # 算os的通信开销
        if comm_range == 1:
            os_comm_cost = 0
        else:
            os_comm_cost = self._get_os_comm_cost(comm_range)

        # 总的通信开销
        comm_cost = os_comm_cost + overlap_cost

        return comm_cost

    def get_comm_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                # TODO：这里需要支持更多的切分策略
                if j != 1 and j % 2 != 0:  # 节点数为奇数的时候
                    self.C[i][j] = self.C[i][j - 1] * 1.2
                else:  # 节点数为偶数
                    self.C[i][j] = self._comm_cost(i, j)

    def _mem_cost(self, i: int, j: int):
        if i == 0:
            if j == 0:
                # 不切P
                return self._dtype_size * self._model_size
            # 对P切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        elif i == 1:
            if j == 0:
                # 不切G
                return self._dtype_size * self._model_size
            # 对G切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        else:
            if j == 0:
                # 不切OS
                return self._dtype_size * self._os_size_ratio * 3 * self._model_size
            # 对OS切j*8份
            return self._dtype_size * self._os_size_ratio * 3 * self._model_size / (j * 8)

    def get_mem_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                if j != 1 and j % 2 != 0:
                    self.A[i][j] = self.A[i][j - 1] * 0.8
                else:
                    self.A[i][j] = self._mem_cost(i, j)


class FSP(BaseAlgo):
    def __init__(
        self, config: dict, cost_data: object, model_config: dict, X: dict, C: dict, A: dict, num_strategies: int
    ) -> None:
        super().__init__(config, cost_data, model_config, X, C, A, num_strategies)
        self.algo_type = AlgoType.FSP

    def set_memory_threshold(self):
        self._activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (34 + (5 * self._a * self._sequence_length / self._h))
            / self._SP
        )
        self._memory_threshold = 80 * (1024**3) - self._activation
        if self._memory_threshold < 0:
            print(f"!!!warning!!!: self._memory_threshold: {self._memory_threshold} < 0")
        print(f"activation: {self._activation:.4f} GB")
        return self._memory_threshold

    def _comm_cost(self, i: int, j: int):
        """
        Get communication cost.

        Args:
            i (int): p (i==0), g (i==1), os (i==2)
            j (int): node count

        Returns:
            float: communication cost

        commu cost = fwd + bwd + optimizer

        fwd = sp + wp
        bwd = sp + wp
        optimizer = zp

        其中 wp的通信可以overlap
        """
        # self._SP_comm = self._get_sp_comm_cost(self._SP)

        if j != 0:
            comm_range = j * 8
        else:
            comm_range = 1  # no comm cost

        # 算overlap的通信开销
        overlap_cost = self.overlap_res._get_overlap(comm_range, self._SP, self.algo_type)

        # 算os的通信开销
        if comm_range == 1:
            os_comm_cost = 0
        else:
            os_comm_cost = self._get_os_comm_cost(comm_range)

        # 总的通信开销
        comm_cost = os_comm_cost + overlap_cost

        return comm_cost

    def get_comm_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                # TODO：这里需要支持更多的切分策略
                if j != 1 and j % 2 != 0:  # 节点数为奇数的时候
                    self.C[i][j] = self.C[i][j - 1] * 1.2
                else:  # 节点数为偶数
                    self.C[i][j] = self._comm_cost(i, j)

    def _mem_cost(self, i: int, j: int):
        if i == 0:
            if j == 0:
                # 不切P
                return self._dtype_size * self._model_size
            # 对P切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        elif i == 1:
            if j == 0:
                # 不切G
                return self._dtype_size * self._model_size
            # 对G切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        else:
            if j == 0:
                # 不切OS
                return self._dtype_size * self._os_size_ratio * 3 * self._model_size
            # 对OS切j*8份
            return self._dtype_size * self._os_size_ratio * 3 * self._model_size / (j * 8)

    def get_mem_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                if j != 1 and j % 2 != 0:
                    self.A[i][j] = self.A[i][j - 1] * 0.8
                else:
                    self.A[i][j] = self._mem_cost(i, j)
