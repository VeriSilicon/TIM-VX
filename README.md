

# TIM-VX - Tensor Interface Module for OpenVX
![VSim.X86.UnitTest](https://github.com/VeriSilicon/TIM-VX/actions/workflows/x86_vsim_unit_test.yml/badge.svg)

TIM-VX is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on OpenVX enabled ML accelerators. It serves as the backend binding for runtime frameworks such as Android NN, Tensorflow-Lite, MLIR, TVM and more.

Main Features
 - Over [130 operators](https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/ops/README.md) with rich format support for both quantized and floating point
 - Simplified C++ binding API calls to create Tensors and Operations
 - Dynamic graph construction with support for shape inference and layout inference
 - Built-in custom layer extensions
 - A set of utility functions for debugging

## Framework Support

- [Tensorflow-Lite](https://github.com/VeriSilicon/tflite-vx-delegate) (External Delegate)
- [Tengine](https://github.com/OAID/Tengine) (Official)
- [TVM](https://github.com/VeriSilicon/tvm) (Fork)
- MLIR Dialect (In development)

Feel free to raise a github issue if you wish to add TIM-VX for other frameworks.

## Get started

### Build and Run
TIM-VX supports both [bazel](https://bazel.build) and cmake. [Install bazel](https://docs.bazel.build/versions/master/install.html) to get started.

TIM-VX needs to be compiled and linked against VeriSilicon OpenVX SDK which provides related header files and pre-compiled libraries. A default linux-x86_64 SDK is provided which contains the simulation environment on PC. Platform specific SDKs can be obtained from respective SoC vendors.

To build TIM-VX
```shell
bazel build libtim-vx.so
```

To run sample LeNet
```shell
# set VIVANTE_SDK_DIR for runtime compilation environment
export VIVANTE_SDK_DIR=`pwd`/prebuilt-sdk/x86_64_linux

bazel build //samples/lenet:lenet_asymu8_cc
bazel run //samples/lenet:lenet_asymu8_cc
```

To build and run Tensorflow-Lite with TIM-VX, please see [README](https://github.com/VeriSilicon/tflite-vx-delegate#readme)
To build and run TVM with TIM-VX, please see [TVM](https://github.com/VeriSilicon/tvm)
