#pragma OPENCL EXTENSION cl_viv_vx_extension : enable

_viv_uniform uint width0;
_viv_uniform uint height0;
_viv_uniform uint width1;
_viv_uniform uint height1;
_viv_uniform uint width_out;
_viv_uniform uint height_out;

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

#define GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(name, data_type, data_type_ptr, stride) \
__kernel void gather_elements_beyond_maxwidth_axis0_##name##_I32to##name \
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
    Tensor index_tensor = create_tensor_from_image2d_array(input1, 4); \
    int* index_ptr = (int*)index_tensor.ptr; \
    int index = index_ptr[coord.x + coord.y * width1 + coord.z * width1 * height1]; \
 \
    Tensor input_tensor = create_tensor_from_image2d_array(input0, stride); \
    data_type_ptr input_ptr = (data_type_ptr)input_tensor.ptr; \
    data_type data = input_ptr[index + coord.y * width0 + coord.z * width0 * height0]; \
 \
    Tensor output_tensor = create_tensor_from_image2d_array(output, stride); \
    data_type_ptr output_ptr = (data_type_ptr)output_tensor.ptr; \
    output_ptr[coord.x + coord.y * width_out + coord.z * width_out * height_out] = data; \
}
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(F32, float, float*, 4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(I32, int,   int*,   4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(F16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(I16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(I8,  char,  char*,  1)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0(U8,  uchar, uchar*, 1)

#define GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(name, data_type, data_type_ptr, stride) \
__kernel void gather_elements_beyond_maxwidth_axis1_##name##_I32to##name \
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
    Tensor index_tensor = create_tensor_from_image2d_array(input1, 4); \
    int* index_ptr = (int*)index_tensor.ptr; \
    int index = index_ptr[coord.x + coord.y * width1 + coord.z * width1 * height1]; \
 \
    Tensor input_tensor = create_tensor_from_image2d_array(input0, stride); \
    data_type_ptr input_ptr = (data_type_ptr)input_tensor.ptr; \
    data_type data = input_ptr[coord.x + index * width0 + coord.z * width0 * height0]; \
 \
    Tensor output_tensor = create_tensor_from_image2d_array(output, stride); \
    data_type_ptr output_ptr = (data_type_ptr)output_tensor.ptr; \
    output_ptr[coord.x + coord.y * width_out + coord.z * width_out * height_out] = data; \
}
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(F32, float, float*, 4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(I32, int,   int*,   4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(F16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(I16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(I8,  char,  char*,  1)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1(U8,  uchar, uchar*, 1)

#define GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(name, data_type, data_type_ptr, stride) \
__kernel void gather_elements_beyond_maxwidth_axis2_##name##_I32to##name \
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
    Tensor index_tensor = create_tensor_from_image2d_array(input1, 4); \
    int* index_ptr = (int*)index_tensor.ptr; \
    int index = index_ptr[coord.x + coord.y * width1 + coord.z * width1 * height1]; \
 \
    Tensor input_tensor = create_tensor_from_image2d_array(input0, stride); \
    data_type_ptr input_ptr = (data_type_ptr)input_tensor.ptr; \
    data_type data = input_ptr[coord.x + coord.y * width0 + index * width0 * height0]; \
 \
    Tensor output_tensor = create_tensor_from_image2d_array(output, stride); \
    data_type_ptr output_ptr = (data_type_ptr)output_tensor.ptr; \
    output_ptr[coord.x + coord.y * width_out + coord.z * width_out * height_out] = data; \
}
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(F32, float, float*, 4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(I32, int,   int*,   4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(F16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(I16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(I8,  char,  char*,  1)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS2(U8,  uchar, uchar*, 1)


#define GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(name, data_type, data_type_ptr, stride) \
__kernel void gather_elements_beyond_maxwidth_axis0_##name##_I32to##name##_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float           input_scale, \
                 float           input_tail, \
                 int             axis_size \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
    Image index_img = create_image_from_image2d(input1, 4); \
    int* index_ptr = (int*)index_img.ptr; \
    int index = index_ptr[coord.x + coord.y * width1]; \
 \
    Image input_img = create_image_from_image2d(input0, stride); \
    data_type_ptr input_ptr = (data_type_ptr)input_img.ptr; \
    data_type data = input_ptr[index + coord.y * width0]; \
 \
    Image output_img = create_image_from_image2d(output, stride); \
    data_type_ptr output_ptr = (data_type_ptr)output_img.ptr; \
    output_ptr[coord.x + coord.y * width_out] = data; \
}
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(F32, float, float*, 4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(I32, int,   int*,   4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(F16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(I16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(I8,  char,  char*,  1)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS0_2D(U8,  uchar, uchar*, 1)

#define GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(name, data_type, data_type_ptr, stride) \
__kernel void gather_elements_beyond_maxwidth_axis1_##name##_I32to##name##_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float           input_scale, \
                 float           input_tail, \
                 int             axis_size \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
    Image index_img = create_image_from_image2d(input1, 4); \
    int* index_ptr = (int*)index_img.ptr; \
    int index = index_ptr[coord.x + coord.y * width1]; \
 \
    Image input_img = create_image_from_image2d(input0, stride); \
    data_type_ptr input_ptr = (data_type_ptr)input_img.ptr; \
    data_type data = input_ptr[coord.x + index  * width0]; \
 \
    Image output_img = create_image_from_image2d(output, stride); \
    data_type_ptr output_ptr = (data_type_ptr)output_img.ptr; \
    output_ptr[coord.x + coord.y * width_out] = data; \
}
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(F32, float, float*, 4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(I32, int,   int*,   4)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(F16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(I16, short, short*, 2)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(I8,  char,  char*,  1)
GATHER_ELEMENTS_BEYOND_MAXWIDTH_AXIS1_2D(U8,  uchar, uchar*, 1)
