# Basic Concepts

## Context

The OpenVX context is the object domain for all OpenVX objects. All data objects live in the context as well as all framework objects. The OpenVX context keeps reference counts on all objects and must do garbage collection during its deconstruction to free lost references.

## Node

A node is an instance of a kernel that will be paired with a specific set of references (the parameters). Nodes are created from and associated with a single graph only.

**Node is also known as Operation.**

## Graph

A set of nodes connected in a directed (only goes one-way) acyclic (does not loop back) fashion. A Graph may have sets of Nodes that are unconnected to other sets of Nodes within the same Graph.

**One context may have several graphs.**

## Tensor

An multidimensional data object.

TIM-VX support the following tensor type:

- Normal Tensor. The tensor which is allocated in host memory, which is managed by OpenVX driver. Normal Tensor is usually used for input/output tensors of the graph.
- Viratul Tensor. The tensor which is allocated in target memory. Viratul Tensor is usually used for intermediate tensors of the graph. Viratul Tensor is faster than the Normal Tensor due to reduce the memory copy between host and target.
- Handle Tensor. The tensor which is allocated in host memory, which is managed by user application. App usually use mmap() to allocate continuous memory for tensor usage. Handle Tensor is also usually used for input/output tensors of the graph. Sometimes the memory of Normal Tensor will be released/relocated by OpenVX driver, and this may be inefficient. You can use Handle Tensor to avoid release/relocate memory.

## Memory Layout

There are two different memory layout mode:

- Row-Major. Also called C contiguous, which is firstly introduced in C by AT&T Bell Lab(1960s). Caffe/TensorFlow/Pytorch follow the Row-Major layout mode.

- Column-Major. Also called Fortran contiguous, which is firstly introduced in Fortran by IBM(1958). Matlab/OpenGL/OpenCL/OpenVX follow the Column-Major layout mode.

See also :

http://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html

On the other hand, a memory layout can be described by both Row-Major and Column-Major mode.

It easily tranlate the layout mode from one to another: just reverse the dimesions of the tensor.

For example, TensorFlow usually describe the tensor as NHWC format. This is equal the CWHN format in OpenVX.

Likewise, the WHCN format in OpenVX is equal the NCHW format in Caffe.

## Data Types

TIM-VX support the following data type:

- INT8
- INT16
- INT32
- INT64
- UINT8
- UINT16
- UINT32
- UINT64
- FLOAT16
- FLOAT32
- FLOAT64

# Graph lifecycle

![Graph lifecycle](/docs/image/graph_lifecycle.png)

- Graph Construction: Create Node/Tensor/Graph.

- Graph Verification: The graphs are checked for consistency, correctness, and other conditions.Memory allocation may occur.

- Graph Execution: Between executions data may be updated by the client or some other external mechanism.

- Graph Deconstruction: Release Node/Tensor/Graph.

# Software architecture

![Software architecture](/docs/image/architecture.png)

- Ovxlib: the C wrapper for OpenVX driver.

- TIM-VX: the C++ wrapper for OpenVX driver.
