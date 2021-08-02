
#define COMPARISONS_F32(func_name, comp_op) \
__kernel void func_name##_F32F32toBOOL8 \
    ( \
    __read_only  image2d_array_t input0, \
    __read_only  image2d_array_t input1, \
    __write_only image2d_array_t output, \
                 float           input0Scale, \
                 float           input0Tail, \
                 float           input1Scale, \
                 float           input1Tail \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    float4 src0; \
    float4 src1; \
    READ_IMAGEF_2DARRAY(src0, input0, coord); \
    READ_IMAGEF_2DARRAY(src1, input1, coord); \
 \
    int4 dst = (src0)comp_op(src1); \
    dst &= 1; \
 \
    write_imagei(output, coord, dst); \
}
COMPARISONS_F32(less, <)
COMPARISONS_F32(great, >)
COMPARISONS_F32(less_equal, <=)
COMPARISONS_F32(great_equal, >=)
COMPARISONS_F32(equal, ==)
COMPARISONS_F32(not_equal, !=)

#define COMPARISONS_F32_2D(func_name, comp_op) \
__kernel void func_name##_F32F32toBOOL8_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float     input0Scale, \
                 float     input0Tail, \
                 float     input1Scale, \
                 float     input1Tail \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src0 = read_imagef(input0, coord); \
    float4 src1 = read_imagef(input1, coord); \
 \
    int4 dst = (src0)comp_op(src1); \
    dst &= 1; \
 \
    write_imagei(output, coord, dst); \
}
COMPARISONS_F32_2D(less, <)
COMPARISONS_F32_2D(great, >)
COMPARISONS_F32_2D(less_equal, <=)
COMPARISONS_F32_2D(great_equal, >=)
COMPARISONS_F32_2D(equal, ==)
COMPARISONS_F32_2D(not_equal, !=)

#define COMPARISONS_U32(func_name, comp_op) \
__kernel void func_name##_U32U32toBOOL8 \
    ( \
    __read_only  image2d_array_t input0, \
    __read_only  image2d_array_t input1, \
    __write_only image2d_array_t output, \
                 float           input0Scale, \
                 float           input0Tail, \
                 float           input1Scale, \
                 float           input1Tail \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    uint4 data0; \
    uint4 data1; \
    READ_IMAGEUI_2DARRAY(data0, input0, coord); \
    READ_IMAGEUI_2DARRAY(data1, input1, coord); \
 \
    float4 src0 = convert_float4(data0) * input0Scale - input0Tail; \
    float4 src1 = convert_float4(data1) * input1Scale - input1Tail; \
    int4 dst = (src0)comp_op(src1); \
    dst &= 1; \
 \
    write_imagei(output, coord, dst); \
}
COMPARISONS_U32(less, <)
COMPARISONS_U32(great, >)
COMPARISONS_U32(less_equal, <=)
COMPARISONS_U32(great_equal, >=)
COMPARISONS_U32(equal, ==)
COMPARISONS_U32(not_equal, !=)

#define COMPARISONS_U32_2D(func_name, comp_op) \
__kernel void func_name##_U32U32toBOOL8_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float     input0Scale, \
                 float     input0Tail, \
                 float     input1Scale, \
                 float     input1Tail \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    uint4 data0 = read_imageui(input0, coord); \
    uint4 data1 = read_imageui(input1, coord); \
 \
    float4 src0 = convert_float4(data0) * input0Scale - input0Tail; \
    float4 src1 = convert_float4(data1) * input1Scale - input1Tail; \
    int4 dst = (src0)comp_op(src1); \
    dst &= 1; \
 \
    write_imagei(output, coord, dst); \
}
COMPARISONS_U32_2D(less, <)
COMPARISONS_U32_2D(great, >)
COMPARISONS_U32_2D(less_equal, <=)
COMPARISONS_U32_2D(great_equal, >=)
COMPARISONS_U32_2D(equal, ==)
COMPARISONS_U32_2D(not_equal, !=)

#define COMPARISONS_I32(func_name, comp_op) \
__kernel void func_name##_I32I32toBOOL8 \
    ( \
    __read_only  image2d_array_t input0, \
    __read_only  image2d_array_t input1, \
    __write_only image2d_array_t output, \
                 float           input0Scale, \
                 float           input0Tail, \
                 float           input1Scale, \
                 float           input1Tail \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    int4 src0; \
    int4 src1; \
    READ_IMAGEI_2DARRAY(src0, input0, coord); \
    READ_IMAGEI_2DARRAY(src1, input1, coord); \
 \
    int4 dst = (src0)comp_op(src1); \
    dst &= 1; \
 \
    write_imagei(output, coord, dst); \
}
COMPARISONS_I32(less, <)
COMPARISONS_I32(great, >)
COMPARISONS_I32(less_equal, <=)
COMPARISONS_I32(great_equal, >=)
COMPARISONS_I32(equal, ==)
COMPARISONS_I32(not_equal, !=)

#define COMPARISONS_I32_2D(func_name, comp_op) \
__kernel void func_name##_I32I32toBOOL8_2D \
    ( \
    __read_only  image2d_t input0, \
    __read_only  image2d_t input1, \
    __write_only image2d_t output, \
                 float     input0Scale, \
                 float     input0Tail, \
                 float     input1Scale, \
                 float     input1Tail \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    int4 src0 = read_imagei(input0, coord); \
    int4 src1 = read_imagei(input1, coord); \
 \
    int4 dst = (src0)comp_op(src1); \
    dst &= 1; \
 \
    write_imagei(output, coord, dst); \
}
COMPARISONS_I32_2D(less, <)
COMPARISONS_I32_2D(great, >)
COMPARISONS_I32_2D(less_equal, <=)
COMPARISONS_I32_2D(great_equal, >=)
COMPARISONS_I32_2D(equal, ==)
COMPARISONS_I32_2D(not_equal, !=)

