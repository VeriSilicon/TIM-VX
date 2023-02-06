- [Extend tim-vx with customized operator](#extend-tim-vx-with-customized-operator)
- [User stories](#user-stories)
- [Design overview](#design-overview)
  - [**Composed operator**](#composed-operator)
    - [Layout Inference {todo}](#layout-inference-todo)
  - [**Customized opencl operator**](#customized-opencl-operator)
    - [How to determine parameter list in a tuple](#how-to-determine-parameter-list-in-a-tuple)
    - [How to initialize](#how-to-determine-parameter-list-in-a-tuple) [custom operation](#extend-tim-vx-with-customized-operator)
    - [How to complete custom operation functions](#how-to-determine-parameter-list-in-a-tuple)
    - [Layout Inference {todo}](#layout-inference-todo-1)

# Extend tim-vx with customized operator

tim-vx will provide two different approches supporting extend AI operators besides built-in ops.
* Compose new operation with builtin ops. example: RNNCell
* Register  opencl kernel as customized operator

# User stories
As **application developer**, I want to **be able to create new opeartor with built-in ops**, so that I can **simplify the lowing from high-level framework(tensorflow,pytorch) to tim-vx**, since I don't want to rewrite same pattern in different frameworks.

As **application developer**, I want to **be able to create my own opeartor with standard opencl kernel**, so that I can **support novel operators not presented in tim-vx**.

# Design overview
![extend.tim-vx.operators](image/extend.tim-vx.operators.png)
* Green components implemented as a public API of tim-vx.
* Red components could be implemented outside of tim-vx.
* Gray components implemented as a private code inside tim-vx.

## **Composed operator**

If some operator can be composed by built-in operators, such as RNNCell which actually built from FullyConnected, Tanh, and DataConvert Layers,
developer can add their own operator implementation before VSI introduce high-performance built-in ops.

[Implementation reference of RNNCell](https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/ops/rnn_cell.cc)

**Keynotes for RNNCell**:

In the constructor of RNNCellImpl, internal operators - fc/tanh/dataconvert - will be created without inner connection.
The inner connection build up inside bindInput() and bindOutput();

### Layout Inference {todo}

Inside of composed operator, it actually is a subgraph of tim-vx's built-in operatos, it should be easy to extend the original layout inference for build-in operators to composed operator - just do layout inference inside the subgraph.

```c++
void ComposedOp::OnInputs(std::vector<std::shared_ptr<vx::Tensor> next_tensor) {
    for(auto op: op_->OpsInSubgraph()) {
        auto Cloned = handleLayoutInference(new_graph, op);
    }
}
```

## **Customized opencl operator**

Customzied kernel should implemented with standard OpenCL 2.0; With tim-vx built-in infrastructure, user can inject their operator with :

1. OpenCL kernel stream as source code;
2. Kernel enqueue configuration for global_work_size and local_work_size;
3. Scalar parameter list defined as a std::tuple;
4. Readable operator name;

TIM-VX provide two different approach to integrate user's operator:
1. Build from source : build tim-vx source and user operators' implementation as single library;
2. Build from sdk: tim-vx prebuilt as a standalone library and a set of standard headers; user build operator implementation and link with tim-vx;

From tim-vx api view, the customized operator registed at graph-level, the registration automatically effected at the first time to create instance of the customized operator. With this approcah, user can override built-in operator or support new operator in a new model easily.

```c++
void CreateGraphWithCustomizedOperator() {
    // create context/graph/tensor as before.
    auto conv = graph->CreateOperation<tim::vx::Conv2d>(...);
    auto post_detect = graph->CreateOperation<3rd_party::DetectionPostProcess>(...);
    post_detect.BindInput(...);
    post_detect.BindOutput(...);

    graph->Compile();
}
```

### How to determine parameter list in a tuple
Usually, kernel take two different kinds of paramter: "tensor-like" and scalar; The tensor-like parameters usually is the output-tensor from other operators or input for other operator.
In the operator's paramter list, only scalar parameters should be defined. "tensor-like" operand should provied by bindInput/bindOutput.

The scalar paramters **MUST** provided at kernel registration.

Take following hswish as example:
CL kernel signature:
```cl
__kernel void hswish_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
```

### How to initialize custom operation
The custom operation class can be defined as:
```c++
  CustomOpClass(Graph*graph, ParamTuple tuple_list, uint32_t input_num,uint32_t output_num)
    : CustomOpBase(graph, input_num, output_num, CustomOpClass::kernel_id_, CustomOpClass::kernel_name_,.../*any other parameter required by c++ code, not relevant to cl kernel**/){
    	tuple_list_.swap(tuple_list);
    	param_transform(tuple_list_, param_list_);
	kernel_resource_="...";
 protected:
	ParamTuple tuple_list_;
  	static const char* kernel_name_;
  	static int32_t kernel_id_;
}
```

1.ParamTuple tuple_list_: scalar parameters tuple list in CL kernel signature, we provide param_transform() function to transform tuple_list_ to param_list_.
2.uint32_t input_num/output_num: the number of kernel operation inputs/outputs.
3.static const char* kernel_name_: OpenCL kernel name defined by users, which is unique.
4.static int32_t kernel_id_:OpenCL kernel id is defined as

```c++
 int32_t CustomOpClass::kernel_id_ = -1 * (++gobal_kernel_id_).
```

5.const char* kernel_resource_: OpenCL kernel registration should be defined in custom op class initialization function. It can contain multi functions adaptd to servel situations. For example:

```c++
kernel_resource_ = "__kernel void hswish_BF16toBF16(\n\
    __read_only  image2d_array_t  input,\n\
    __write_only image2d_array_t  output,\n\
                           float  beta\n\
    )\n\
{\n\
   /*kernel funtion resource*/\n\
}\n\
\n\
__kernel void hswish_BF32toF32(\n\
    __read_only  image2d_array_t  input,\n\
    __write_only image2d_array_t  output,\n\
                           float  inputScale,\n\
                           float  inputTail,\n\
                           float  outputScale,\n\
                           float  outputZP\n\
    )\n\
{\n\
  /*kernel funtion resource*/ \n\
}\n\";
```

### How to complete custom operation functions

1.SetupShapeInfor: the function for output tensor size.
```c++
void SetupShapeInfor() override {
   outputs_size_[0].push_back(...);
   ...
}
```

2.SetupParams:  the function for kernel select and build option. The func_name_ is the selected function name provided by kernel_resource_, is used to determine which kernel function to be applied. build_option is the compiler options when compile custom op resource.
```c++
  void SetupParams(
      std::vector<tim::vx::DataType> input_types,
      std::string& build_option) override {
      if(...){
        func_name_ = "..."/*it MUST provided in kernel_source_ */;
        build_option = "..."/*compile paramters*/;
      }else{
	...
      }
  }
```

3.SetupEnqueue: the function for kernel local size and gobal size.
```c++
void SetupEnqueue(uint32_t& dim, std::vector<size_t>& global_size,
                  std::vector<size_t>& local_size) {
  dim = .../*kernel dim*/;
  local_size[0] = .../*kernel local size*/;
  global_size[0] = .../*kernel global size*/;
}
```
local_size and global_size are similar features as **clEnqueueNDRangeKernel** in standard OpenCL.

Some tips for work_size:
    HWThreadCount = 4

4.Clone: the function for operation clone.
```c++
std::shared_ptr<tim::vx::Operation> Clone(
      std::shared_ptr<tim::vx::Graph>& graph) const override{
   return graph->CreateOperation<user::custom_operation>(graph,this->params/*others*/);
}
```

### Layout Inference

so far we don't support this feature. User should take care of the layout transform carefully.
TODO: vsi will rework the framework so that any customized op can work properly in layout transform.
