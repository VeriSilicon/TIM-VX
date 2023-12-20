# ApiTracer - Header only Cpp OO programs trace and replay tool

ApiTracer is a header only library provides macros and template functions to trace the C++ object-oriented programs, it not only trace the call-stacks, but also trace the C++ Apis runtime parameters. With the trace log and binary, it's convenient to replay the program execute scene, which helps developer to reproduce bugs and debug.

## Usage
There is a simple case using api trace in `src/tim/vx/graph_test.cc` which named trace_test, follow those steps:  
1. You need set the cmake option `-DTIM_VX_ENABLE_API_TRACE=1` at first, and rebuild;
2. Setting up the run time environments for tim-vx, then execute `$build/install/bin/unit-test --gtest_filter=*trace_test*`;
3. `trace_log.cc` and `trace_bin.bin` will generate in the workspace, first one is the tim-vx code record the all api calls and function's runtime parameters, second one is serializable data like std::vector data. You can set the environment `TRACE_DUMP_PREFIX` to control will to dump those file;
4. Copy `trace_log.cc` to the root of tim-vx source code, and rename it with `trace_log.rpl.cc`, then follow the guide in `src/tim/vx/graph_test.cc`: test case - **replay_test**.
5. Rebuild unit-test, execute `$build/install/bin/unit-test --gtest_filter=*replay_test*`, you will get the exactly same result with traced execution.
**Caution**: if your boost library version lower than 1.61.0, you can't compile tracer because lack of boost.hana library, but you can still comiple the replayer code(disable ENABLE_API_TRACE).

## Coding work
ApiTracer was implemented by warp original Apis in traced Apis, they got same class names, same function names, but different namespace. So you need implement the traced Apis for specific programs at first.

### Examples of implement traced class
Original TensorSpec:
``` C++
namespace tvx {
struct TensorSpec {
  TensorSpec() {}
  TensorSpec(DataType datatype, const ShapeType& shape, TensorAttribute attr)
      : datatype_(datatype), shape_(shape), attr_(attr) {}

  TensorSpec(DataType datatype, const ShapeType& shape, TensorAttribute attr,
             const Quantization& quantization)
      : TensorSpec(datatype, shape, attr) {
    this->quantization_ = quantization;
  }

  TensorSpec(const TensorSpec& other);

  TensorSpec& operator=(const TensorSpec& other);

  TensorSpec& SetDataType(DataType datatype);

  TensorSpec& SetShape(const ShapeType& shape);

  TensorSpec& SetAttribute(TensorAttribute attr);
  ...
};

} /* namespace tvx */
```
Define traced TensorSpec
``` C++
namespace trace {
// all traced classes should derive from TraceClassBase<T>
struct TensorSpec : public TraceClassBase<target::TensorSpec> {
  DEF_CONSTRUCTOR(TensorSpec)

  // define constructor with boost seq arguments, do not need specific the
  // parameter name, it's auto generate like param_0, param_1, ...
  DEF_CONSTRUCTOR(TensorSpec, ((DataType))
                              ((const ShapeType&))
                              ((TensorAttribute))
  )

  DEF_CONSTRUCTOR(TensorSpec, ((DataType))
                              ((const ShapeType&))
                              ((TensorAttribute))
                              ((const Quantization&))
  )

  DEF_CONSTRUCTOR(TensorSpec, ((const TensorSpec&))
  )

  // Using DEF_TRACED_API is a more radical way, you only need specific the api
  // return type and the api name, api arguments will automatically match by the
  // template arguments variadic.
  // But it's not fully around now, see next section of it's restrict.
  DEF_TRACED_API(TensorSpec&, operator=)

  DEF_TRACED_API(TensorSpec&, SetDataType)

  DEF_TRACED_API(TensorSpec&, SetShape)

  DEF_TRACED_API(TensorSpec&, SetAttribute)
  ...
};
}
```

### Macros definition
- DEF_CONSTRUCTOR(class_name, optional_arguments_description, optional_macro_to_handle_pointer):  
arguments_description must follow the format `((arg0))((arg1)(arg1_default_value))`

- DEF_TRACED_API(return_type, member_function_name, optional_lambda_to_handle_pointer):  
this macro can define most member functions, except:
  - function with default parameters
  - static functions
  - readonly functions

- DEF_MEMFN_SP(return_type_removed_shared_pointer, member_function_name, optional_arguments_description, optional_macro_to_handle_pointer):  
the macro can define those functions with `shared_ptr<TracedClass>`

- DEF_MEMFN(return_type, member_function_name, optional_arguments_description, optional_macro_to_handle_pointer):  
to define normal member functions

- DEF_INPLACE_MEMFN(member_function_name, optional_arguments_description, optional_macro_to_handle_pointer):  
to define those apis retval is the object itself

- DEF_SIMPLE_UNTRACED_API(retval, api_name):  
to define those apis without argument and have no side effect on the graph compile and execution, like `Tensor::GetId()`

## Meta programming reference
api trace utilize some meta programming tricks, here are some reference:
1. Boost C/C++ preprocessor - https://www.boost.org/doc/libs/1_82_0/libs/preprocessor/doc/index.html
2. std meta programming library - https://en.cppreference.com/w/cpp/meta
3. Boost hana meta programming library - https://www.boost.org/doc/libs/1_82_0/libs/hana/doc/html/index.html
