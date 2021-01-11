# TIM-VX - Tensor Interface Module for OpenVX

TIM-VX is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on OpenVX enabled ML accelerators. It serves as the backend binding for runtime frameworks such as Android NN, Tensorflow-Lite, MLIR, TVM and more.

Main Features
 - Over 130 internal operators with rich format support for both quantized and floating point
 - Simplified binding API calls to create Tensors and Operations
 - Dynamic graph construction and supports shape inferencing
 - Built-in custom layer extensions
 - A set of utility functions for debugging

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
bazel build //samples/lenet:lenet_asymu8_cc
bazel run //samples/lenet:lenet_asymu8_cc
```

### Get familiar with OpenVX spec
To development for TIM-VX, you first need to get familiar with [OpenVX API](https://www.khronos.org/openvx/) and [OpenVX NN Extension API](https://www.khronos.org/registry/vx). Please head over to [Khronos](https://www.khronos.org/) to read the spec.

