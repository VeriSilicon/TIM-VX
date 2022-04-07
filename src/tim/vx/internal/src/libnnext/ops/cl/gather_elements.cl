
#define GATHER_ELEMENTS_AXIS0_2D(name, data_type, read_func, write_func, conv_func) \
__kernel void gather_elements_axis0_##name##_I32to##name##_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float     input_scale, \
                 float     input_tail, \
                 int       axis_size \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
 \
    int index = read_imagei(input1, coord).x; \
    int index1 = index + axis_size; \
    index = index < 0 ? index1 : index; \
 \
    data_type data = read_func(input0, (int2)(index, coord.y)); \
    float4 dst = convert_float4(data) * input_scale + input_tail; \
    data = conv_func(dst); \
 \
    write_func(output, coord, data); \
}
GATHER_ELEMENTS_AXIS0_2D(F32, float4, read_imagef,  write_imagef,  convert_float4)
GATHER_ELEMENTS_AXIS0_2D(I32, int4,   read_imagei,  write_imagei,  convert_int4_rte)
GATHER_ELEMENTS_AXIS0_2D(U32, uint4,  read_imageui, write_imageui, convert_uint4_rte)

#define GATHER_ELEMENTS_AXIS0(name, data_type, read_func, write_func, conv_func) \
__kernel void gather_elements_axis0_##name##_I32to##name \
    ( \
    __read_only  image2d_array_t input0, \
    __read_only  image2d_array_t input1, \
    __write_only image2d_array_t output, \
                 float           input_scale, \
                 float           input_tail, \
                 int             axis_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2)); \
 \
    int index = read_imagei(input1, coord).x; \
    int index1 = index + axis_size; \
    index = index < 0 ? index1 : index; \
 \
    data_type data = read_func(input0, (int4)(index, coord.yzz)); \
    float4 dst = convert_float4(data) * input_scale + input_tail; \
    data = conv_func(dst); \
 \
    write_func(output, coord, data); \
}
GATHER_ELEMENTS_AXIS0(F32, float4, read_imagef,  write_imagef,  convert_float4)
GATHER_ELEMENTS_AXIS0(I32, int4,   read_imagei,  write_imagei,  convert_int4_rte)
GATHER_ELEMENTS_AXIS0(U32, uint4,  read_imageui, write_imageui, convert_uint4_rte)

#define GATHER_ELEMENTS_AXIS1_2D(name, data_type, read_func, write_func, conv_func) \
__kernel void gather_elements_axis1_##name##_I32to##name##_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float     input_scale, \
                 float     input_tail, \
                 int       axis_size \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
 \
    int index = read_imagei(input1, coord).x; \
    int index1 = index + axis_size; \
    index = index < 0 ? index1 : index; \
 \
    data_type data = read_func(input0, (int2)(coord.x, index)); \
    float4 dst = convert_float4(data) * input_scale + input_tail; \
    data = conv_func(dst); \
 \
    write_func(output, coord, data); \
}
GATHER_ELEMENTS_AXIS1_2D(F32, float4, read_imagef,  write_imagef,  convert_float4)
GATHER_ELEMENTS_AXIS1_2D(I32, int4,   read_imagei,  write_imagei,  convert_int4_rte)
GATHER_ELEMENTS_AXIS1_2D(U32, uint4,  read_imageui, write_imageui, convert_uint4_rte)

#define GATHER_ELEMENTS_AXIS1(name, data_type, read_func, write_func, conv_func) \
__kernel void gather_elements_axis1_##name##_I32to##name \
    ( \
    __read_only  image2d_array_t input0, \
    __read_only  image2d_array_t input1, \
    __write_only image2d_array_t output, \
                 float           input_scale, \
                 float           input_tail, \
                 int             axis_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2)); \
 \
    int index = read_imagei(input1, coord).x; \
    int index1 = index + axis_size; \
    index = index < 0 ? index1 : index; \
 \
    data_type data = read_func(input0, (int4)(coord.x, index, coord.zz)); \
    float4 dst = convert_float4(data) * input_scale + input_tail; \
    data = conv_func(dst); \
 \
    write_func(output, coord, data); \
}
GATHER_ELEMENTS_AXIS1(F32, float4, read_imagef,  write_imagef,  convert_float4)
GATHER_ELEMENTS_AXIS1(I32, int4,   read_imagei,  write_imagei,  convert_int4_rte)
GATHER_ELEMENTS_AXIS1(U32, uint4,  read_imageui, write_imageui, convert_uint4_rte)

#define GATHER_ELEMENTS_AXIS2(name, data_type, read_func, write_func, conv_func) \
__kernel void gather_elements_axis2_##name##_I32to##name \
    ( \
    __read_only  image2d_array_t input0, \
    __read_only  image2d_array_t input1, \
    __write_only image2d_array_t output, \
                 float           input_scale, \
                 float           input_tail, \
                 int             axis_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2)); \
 \
    int index = read_imagei(input1, coord).x; \
    int index1 = index + axis_size; \
    index = index < 0 ? index1 : index; \
 \
    data_type data = read_func(input0, (int4)(coord.xy, index, coord.z)); \
    float4 dst = convert_float4(data) * input_scale + input_tail; \
    data = conv_func(dst); \
 \
    write_func(output, coord, data); \
}
GATHER_ELEMENTS_AXIS2(F32, float4, read_imagef,  write_imagef,  convert_float4)
GATHER_ELEMENTS_AXIS2(I32, int4,   read_imagei,  write_imagei,  convert_int4_rte)
GATHER_ELEMENTS_AXIS2(U32, uint4,  read_imageui, write_imageui, convert_uint4_rte)
