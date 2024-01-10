from typing import Any, List
from numpy.typing import NDArray
from argparse import ArgumentParser
from pathlib import Path
from nbg_runner import OVXExecutor
import cv2 as cv
import numpy as np


def get_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument(
        "--nbg", "-m",
        type=Path,
        required=True,
        help="Path to NBG file."
    )
    parser.add_argument(
        "--image", "-i",
        type=Path,
        required=True,
        help="Path to image file."
    )
    parser.add_argument(
        "--labels", "-l",
        type=Path,
        required=True,
        help="Path to classification labels."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    nbg_path: Path = args.nbg
    img_path: Path = args.image
    labels_path: Path = args.labels

    # Load NBG and query I/O params.
    executor = OVXExecutor(nbg_path)
    num_inputs = executor.get_num_inputs()
    num_outputs = executor.get_num_outputs()
    input_info = executor.get_input_info(index=0)
    output_info = executor.get_output_info(index=0)

    input_size = input_info.shape[1:3]
    num_cls = output_info.shape[1]

    # Load input image.
    img_hwc: NDArray[np.uint8] = cv.imread(str(img_path), cv.IMREAD_COLOR)
    hi, wi = input_size
    img_hwc = cv.resize(img_hwc, dsize=(wi, hi))
    img_hwc = cv.cvtColor(img_hwc, cv.COLOR_BGR2RGB)
    img_nhwc = np.expand_dims(img_hwc, axis=0)

    if input_info.dtype == "float32":
        img_nhwc = img_nhwc.astype(np.float32) / np.iinfo(np.uint8).max

    # Load classification labels.
    cls_labels: List[str] = []
    with open(labels_path, mode="r") as f:
        for label in f:
            cls_labels.append(label.strip())

    # Run inference.
    # executor.set_inputs([img_nhwc])
    executor.set_input(index=0, input_tensor=img_nhwc)
    executor.run()
    scores = executor.get_output(index=0)
    # scores = executor.get_outputs()[0]

    cls = np.argmax(scores, axis=1)
    cls = np.squeeze(cls, axis=0).item()

    cls_label = cls_labels[cls]
    print(f"Classification result: {cls_label}")
