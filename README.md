

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

- [Tensorflow-Lite Delegate](https://github.com/VeriSilicon/tensorflow/tree/vx-delegate.v2.4.1) (Unofficial)
- [Tengine](https://github.com/OAID/Tengine) (Official)
- MLIR Dialect (In development)
- TVM (In development)

## Roadmap

Roadmap of TIM-VX will be updated here in the future.

## Get started

### Build and Run
TIM-VX uses [bazel](https://bazel.build) build system by default. [Install bazel](https://docs.bazel.build/versions/master/install.html) first to get started.

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

To build and run Tensorflow-Lite delegate on A311D platform
```shell
# clone and cross build VeriSilicon tensorflow fork with TFlite delegate support
git clone --single-branch --branch vx-delegate.v2.4.1 git@github.com:VeriSilicon/tensorflow.git vx-delegate; cd vx-delegate
bazel build --config A311D //tensorflow/lite/tools/benchmark:benchmark_model

# push benchmark_model onto device and run
./benchmark_model --graph=mobilenet_v1_1.0_224_quant.tflite --use_vxdelegate=true
```
