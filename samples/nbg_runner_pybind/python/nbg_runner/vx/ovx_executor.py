from typing import List, Tuple, Sequence
from numpy.typing import NDArray
import numpy as np
from pathlib import Path
from nbg_runner import _binding


class OVXTensorInfo:
    rank: int = ...
    shape: Tuple[int, ...] = ...
    dtype: str = ...
    qtype: str = ...
    scale: float = ...
    zero_point: int = ...
    fixed_point_pos: int = ...


class OVXExecutor:
    def __init__(self, nbg_path: Path) -> None:
        self._exec = _binding.OVXExecutor(nbg_path)
        self._exec.init()

    def get_num_inputs(self) -> int:
        return self._exec.get_num_inputs()

    def get_num_outputs(self) -> int:
        return self._exec.get_num_outputs()

    def get_input_info(self, index: int) -> OVXTensorInfo:
        return self._exec.get_input_info(index)

    def get_output_info(self, index: int) -> OVXTensorInfo:
        return self._exec.get_output_info(index)

    def get_input_infos(self) -> List[OVXTensorInfo]:
        input_infos: List[OVXTensorInfo] = []
        num_inputs = self.get_num_inputs()
        for i in range(num_inputs):
            input_infos.append(self.get_input_info(i))
        return input_infos

    def get_output_infos(self) -> List[OVXTensorInfo]:
        output_infos: List[OVXTensorInfo] = []
        num_outputs = self.get_num_outputs()
        for i in range(num_outputs):
            output_infos.append(self.get_output_info(i))
        return output_infos

    def set_input(self, index: int, input_tensor: NDArray) -> None:
        return self._exec.set_input(index, input_tensor)

    def get_output(self, index: int) -> NDArray:
        output_tensor: NDArray = self._exec.get_output(index)
        return output_tensor

    def set_inputs(self, input_tensors: Sequence[NDArray]) -> None:
        for i, tensor in enumerate(input_tensors):
            self.set_input(i, tensor)

    def get_outputs(self) -> List[NDArray]:
        output_tensors: List[NDArray] = []
        num_outputs = self.get_num_outputs()
        for i in range(num_outputs):
            output_tensor = self.get_output(i)
            output_tensors.append(output_tensor)

        return output_tensors

    def run(self) -> None:
        self._exec.run()
