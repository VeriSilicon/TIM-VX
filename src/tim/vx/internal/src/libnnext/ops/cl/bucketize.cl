#pragma OPENCL EXTENSION CL_VIV_asm : enable

#define BUCKETIZE_F32_2D_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t boundaries, \
    __write_only image2d_t output, \
                 int       boundaries_size, \
                 float     input0_scale, \
                 float     input0_tail, \
                 float     input1_scale, \
                 float     input1_tail \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src0 = read_imagef(input, coord); \
 \
    int2 pos = 0; \
    do \
    { \
        float4 src1 = read_imagef(boundaries, pos); \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_F32_2D_SH_IMPL(F32_F32toI32_2D,       <=)
BUCKETIZE_F32_2D_SH_IMPL(right_F32_F32toI32_2D, <)

#define BUCKETIZE_F32_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_t       boundaries, \
    __write_only image2d_array_t output, \
                 int             boundaries_size, \
                 float           input0_scale, \
                 float           input0_tail, \
                 float           input1_scale, \
                 float           input1_tail \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    float4 src0 = read_imagef(input, coord); \
 \
    int2 pos = 0; \
    do \
    { \
        float4 src1 = read_imagef(boundaries, pos); \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_F32_SH_IMPL(F32_F32toI32,       <=)
BUCKETIZE_F32_SH_IMPL(right_F32_F32toI32, <)

#define BUCKETIZE_I32_2D_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t boundaries, \
    __write_only image2d_t output, \
                 int       boundaries_size, \
                 float     input0_scale, \
                 float     input0_tail, \
                 float     input1_scale, \
                 float     input1_tail \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src0 = convert_float4(read_imagei(input, coord)); \
 \
    int2 pos = 0; \
    src0 = src0 * input0_scale + input0_tail; \
    do \
    { \
        float4 src1 = convert_float4(read_imagei(boundaries, pos)); \
        src1 = src1 * input1_scale + input1_tail; \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_I32_2D_SH_IMPL(I32_I32toI32_2D,       <=)
BUCKETIZE_I32_2D_SH_IMPL(right_I32_I32toI32_2D, <)

#define BUCKETIZE_I32_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_t       boundaries, \
    __write_only image2d_array_t output, \
                 int             boundaries_size, \
                 float           input0_scale, \
                 float           input0_tail, \
                 float           input1_scale, \
                 float           input1_tail \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    int4 data = read_imagei(input, coord); \
    float4 src0 = convert_float4(data) * input0_scale + input0_tail; \
 \
    int2 pos = 0; \
    do \
    { \
        float4 src1 = convert_float4(read_imagei(boundaries, pos)); \
        src1 = src1 * input1_scale + input1_tail; \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_I32_SH_IMPL(I32_I32toI32,       <=)
BUCKETIZE_I32_SH_IMPL(right_I32_I32toI32, <)

#define BUCKETIZE_U32_2D_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t boundaries, \
    __write_only image2d_t output, \
                 int       boundaries_size, \
                 float     input0_scale, \
                 float     input0_tail, \
                 float     input1_scale, \
                 float     input1_tail \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src0 = convert_float4(read_imageui(input, coord)); \
 \
    int2 pos = 0; \
    src0 = src0 * input0_scale + input0_tail; \
    do \
    { \
        float4 src1 = convert_float4(read_imageui(boundaries, pos)); \
        src1 = src1 * input1_scale + input1_tail; \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_U32_2D_SH_IMPL(U32_U32toI32_2D,       <=)
BUCKETIZE_U32_2D_SH_IMPL(right_U32_U32toI32_2D, <)

#define BUCKETIZE_U32_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_t       boundaries, \
    __write_only image2d_array_t output, \
                 int             boundaries_size, \
                 float           input0_scale, \
                 float           input0_tail, \
                 float           input1_scale, \
                 float           input1_tail \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    uint4 data = read_imageui(input, coord); \
    float4 src0 = convert_float4(data) * input0_scale + input0_tail; \
 \
    int2 pos = 0; \
    do \
    { \
        float4 src1 = convert_float4(read_imageui(boundaries, pos)); \
        src1 = src1 * input1_scale + input1_tail; \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_U32_SH_IMPL(U32_U32toI32,       <=)
BUCKETIZE_U32_SH_IMPL(right_U32_U32toI32, <)

#define BUCKETIZE_BF16_2D_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t boundaries, \
    __write_only image2d_t output, \
                 int       boundaries_size, \
                 float     input0_scale, \
                 float     input0_tail, \
                 float     input1_scale, \
                 float     input1_tail \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
 \
    uint4 data0 = read_imageui(input, coord) << 16; \
    float4 src0; \
    _viv_asm(COPY, src0, data0, 16); \
 \
    int2 pos = 0; \
    do \
    { \
        uint4 data1 = read_imageui(boundaries, pos) << 16; \
        float4 src1; \
        _viv_asm(COPY, src1, data1, 16); \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_BF16_2D_SH_IMPL(BF16_BF16toI32_2D,       <=)
BUCKETIZE_BF16_2D_SH_IMPL(right_BF16_BF16toI32_2D, <)

#define BUCKETIZE_BF16_SH_IMPL(name, comp_op) \
__kernel void bucketize_##name \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_t       boundaries, \
    __write_only image2d_array_t output, \
                 int             boundaries_size, \
                 float           input0_scale, \
                 float           input0_tail, \
                 float           input1_scale, \
                 float           input1_tail \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    uint4 data0 = read_imageui(input, coord) << 16; \
    float4 src0; \
    _viv_asm(COPY, src0, data0, 16); \
 \
    int2 pos = 0; \
    do \
    { \
        uint4 data1 = read_imageui(boundaries, pos) << 16; \
        float4 src1; \
        _viv_asm(COPY, src1, data1, 16); \
        if ((src0.x) comp_op (src1.x)) \
        { \
            break; \
        } \
        pos.x ++; \
    } while(pos.x < boundaries_size); \
 \
    write_imagei(output, coord, pos.xxxx); \
}
BUCKETIZE_BF16_SH_IMPL(BF16_BF16toI32,       <=)
BUCKETIZE_BF16_SH_IMPL(right_BF16_BF16toI32, <)
