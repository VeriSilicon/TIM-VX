# ApiTracer - Header only Cpp OO programs trace and replay tool

ApiTracer is a header only library provides macros and template functions to trace the C++ object-oriented programs, it not only trace the call-stacks, but also trace the C++ Apis runtime parameters. With the trace log and binary, it's convenient to replay the program execute scene, which helps developer to reproduce bugs and debug.

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

  TensorSpec& SetShape(ShapeType& shape);

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

- DEF_MEMFN_SP(return_type_removed_shared_pointer, member_function_name,optional_arguments_description, optional_macro_to_handle_pointer):  
the macro can define those functions with `shared_ptr<TracedClass>`
- DEF_MEMFN(return_type, member_function_name, optional_arguments_description, optional_macro_to_handle_pointer)
- DEF_INPLACE_MEMFN(member_function_name, optional_arguments_description, optional_macro_to_handle_pointer)
