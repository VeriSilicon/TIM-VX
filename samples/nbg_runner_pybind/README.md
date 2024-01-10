# VSI NBG Runner Python Binding

This project is a python package that wraps OpenVX API using pybind11. It provides simple python API to load, query and run model NBG.

## Installation

```cmake
cmake -B build -DVIVANTE_SDK_DIR=${VIV_SDK_INSTALL_PATH}
cmake --build build
```

The built python binding lib can be found at
`build/src/_nbg_runner.cpython-{python_version}-{platform}.so`. Place the lib into `python/nbg_runner/_binding/`.

## Usage

### Tensor Info

| Field             | Type              |  Value Sample |
|:-----------------:|:-----------------:|:-------------:|
| rank              | int               | 4             |
| shape             | Tuple[int, ...]   | (1,3,224,224) |
| dtype             | str               | "uint8"       |
| qtype             | str               | "affine"      |
| scale             | float             | 0.007874      |
| zero_point        | int               | 128           |
| fixed_point_pos   | int               | 0             |

- `shape` is in C-style row major order, which is consistent with NumPy.

### Set Environment Vars

```shell
# Set HW target If the driver is compiled with vsimulator.
VSIMULATOR_CONFIG=VIP9000ULSI_PID0XBA
# Locate the OVX driver.
VIVANTE_SDK_DIR=${VIV_SDK_INSTALL_PATH}
LD_LIBRARY_PATH=${VIVANTE_SDK_DIR}/[lib|lib64|drivers]
# Set PYTHONPATH to the dir containing nbg_runner module.
PYTHONPATH=${workspaceFolder}/python
```

### Example

See detailed examples in `examples/*.py`

```python
from nbg_runner import OVXExecutor

# Load a model NBG file.
executor = OVXExecutor("path/to/model.nbg")

# Query model I/O tensors count.
num_inputs = executor.get_num_inputs()
num_outputs = executor.get_num_outputs()

# Get I/O tensor info by index.
input_info = executor.get_input_info(0)
output_info = executor.get_output_info(0)

# Or get all I/O tensors infos at once.
input_infos = executor.get_input_infos()
output_infos = executor.get_output_infos()

# Prepare inputs.
input_tensors: List[NDArray] = ...

# Set input tensor by index.
for i, input_tensor in enumerate(input_tensors):
    executor.set_input(i, input_tensor)

# Or set all input tensors at once.
executor.set_inputs(input_tensors)

# Run inference.
executor.run()

# Get output tensor by index.
for i in range(num_outputs):
    output_tensor = executor.get_output(i)

# Or get all output tensors at once.
output_tensors = executor.get_outputs()
```
