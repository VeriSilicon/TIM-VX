

# TIM-VX - Tensor Interface Module
![Bazel.VSim.X86.UnitTest](https://github.com/VeriSilicon/TIM-VX/actions/workflows/bazel_x86_vsim_unit_test.yml/badge.svg)
![CMake.VSim.X86.UnitTest](https://github.com/VeriSilicon/TIM-VX/actions/workflows/cmake_x86_vsim_unit_test.yml/badge.svg)

TIM-VX is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on Verisilicon ML accelerators. It serves as the backend binding for runtime frameworks such as Android NN, Tensorflow-Lite, MLIR, TVM and more.

Main Features
 - Over [150 operators](https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/ops/README.md) with rich format support for both quantized and floating point
 - Simplified C++ binding API calls to create Tensors and Operations [Guide](https://github.com/VeriSilicon/TIM-VX/blob/main/docs/Programming_Guide.md)
 - Dynamic graph construction with support for shape inference and layout inference
 - Built-in custom layer extensions
 - A set of utility functions for debugging

## Framework Support

- [Tensorflow-Lite](https://github.com/VeriSilicon/tflite-vx-delegate) (External Delegate)
- [Tengine](https://github.com/OAID/Tengine) (Official)
- [TVM](https://github.com/VeriSilicon/tvm) (Fork)
- MLIR Dialect (In development)

Feel free to raise a github issue if you wish to add TIM-VX for other frameworks.

## Architecture Overview

![TIM-VX Architecture](docs/image/timvx_overview.svg)

# Get started

## Build and Run

TIM-VX supports both [bazel](https://bazel.build) and cmake.

### Cmake

To build TIM-VX:

```shell
mkdir host_build
cd host_build
cmake ..
make -j8
make install
```

All install files (both headers and *.so) is located in : `host_build/install`

Cmake option:

`CONFIG`: Set Target Platform. Such as: `A311D`, `S905D3`, `vim3_android`, `YOCTO`. Default is `X86_64_linux`.

`TIM_VX_ENABLE_TEST`: Build the unit test. Default is OFF.

`TIM_VX_USE_EXTERNAL_OVXLIB`: Use external OVXLIB. Default is OFF.

`EXTERNAL_VIV_SDK`: use external VX driver libs. By default is OFF.

run unit test:

```shell
cd host_build/src/tim
export LD_LIBRARY_PATH=`pwd`/../../../prebuilt-sdk/x86_64_linux/lib:$LD_LIBRARY_PATH
./unit_test
```

### Bazel

[Install bazel](https://docs.bazel.build/versions/master/install.html) to get started.

TIM-VX needs to be compiled and linked against VeriSilicon OpenVX SDK which provides related header files and pre-compiled libraries. A default linux-x86_64 SDK is provided which contains the simulation environment on PC. Platform specific SDKs can be obtained from respective SoC vendors.

To build TIM-VX:

```shell
bazel build libtim-vx.so
```

To run sample LeNet:

```shell
# set VIVANTE_SDK_DIR for runtime compilation environment
export VIVANTE_SDK_DIR=`pwd`/prebuilt-sdk/x86_64_linux

bazel build //samples/lenet:lenet_asymu8_cc
bazel run //samples/lenet:lenet_asymu8_cc
```

## Other

To build and run Tensorflow-Lite with TIM-VX, please see [README](https://github.com/VeriSilicon/tflite-vx-delegate#readme)

To build and run TVM with TIM-VX, please see [TVM README](https://github.com/VeriSilicon/tvm/blob/vsi_npu/README.VSI.md)

# Reference board

Chip | Vendor | References 
:------    |:----- |:------
i.MX 8M Plus | NXP | [download BSP](https://www.nxp.com/design/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX?tab=Design_Tools_Tab)

# Support
create issue on github or email to ML_Support@verisilicon.com
