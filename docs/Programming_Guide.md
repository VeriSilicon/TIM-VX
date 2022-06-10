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

TIM-VX support the following tensor attributes:

- INPUT Tensor: A tensor object used as input for the graph. Contents can be updated by the host prior to each inference. INPUT Tensor can be mapped from a host allocated memory handle. 

- OUTPUT Tensor: A tensor object used as output for the graph. Contents can be retribed by the host after each inference. Output Tensor can be mapped from a host allocated memory handle.  

- VARIABLE Tensor: A tensor object which can be used as both input and output for the graph, typically used in recurrent networks to hold recurrent states. It's contents are accessible by the host. Variable Tensor can be mapped from a host allocated memory handle.  

- CONSTANT Tensor: A tensor object that holds contents which will not change throughout the execution of the graph. The contents of these tensor objects must be populated prior to graph compilation, and shall not change afterwards. This type of tensors can be used to hold weights, bias or other types parameters.

- TRANSIENT Tensor: A virtual tensor object which is only used to represent the connectivity between nodes, and its contents are not accessible by the host. The life cycle of these tensors are managed entirely by TIM-VX internally, and they are mostly used to represent intermediate activations of a graph. 

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
- UINT8
- UINT16
- UINT32
- FLOAT16
- FLOAT32

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
