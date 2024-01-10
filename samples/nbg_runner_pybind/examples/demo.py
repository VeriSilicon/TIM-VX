from typing import Any
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from nbg_runner import OVXExecutor


def get_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument(
        "--nbg", "-m",
        type=Path,
        default="examples/models/conv2d_relu_maxpool2d_fp32.nbg",
        help="Path to NBG file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    nbg_path: Path = args.nbg

    executor = OVXExecutor(nbg_path)
    num_inputs = executor.get_num_inputs()
    num_outputs = executor.get_num_outputs()

    input_info = executor.get_input_info(0)
    output_info = executor.get_output_info(0)

    input_tensor = np.ones(
        shape=input_info.shape,
        dtype=input_info.dtype
    )

    executor.set_input(0, input_tensor)
    executor.run()
    output_tensor = executor.get_output(0)
    print(output_tensor)
