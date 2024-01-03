float sigmoid_(float x, float logE)
{
    x *= -logE;
    x = 1 + exp2(x);
    return 1 / x;
}

#define SWISH_F32_F32_PROCESS() \
    float4 src, tmp, dst; \
    src   = read_imagef(input, coord); \
    tmp.x = sigmoid_(src.x * beta, logE); \
    dst.x = src.x * tmp.x; \
    write_imagef(output, coord, dst);

__kernel void swish_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    SWISH_F32_F32_PROCESS()
}

__kernel void swish_F32toF32_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    SWISH_F32_F32_PROCESS()
}


#define SWISH_U8_U8_PROCESS() \
    float4 src, tmp, data; \
    uint4 src0 = read_imageui(input, coord); \
    src   = convert_float4(src0) * inputScale - inputTail; \
    tmp.x = sigmoid_(src.x * beta, logE); \
    data.x = src.x * tmp.x; \
    uint4 dst = convert_uint4_rte(data * outputScale + outputZP); \
    write_imageui(output, coord, dst);

__kernel void swish_U8toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    SWISH_U8_U8_PROCESS()
}

__kernel void swish_U8toU8_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    SWISH_U8_U8_PROCESS()
}


#define SWISH_I32_I32_PROCESS() \
    float4 src, tmp, data; \
    int4 src0 = read_imagei(input, coord); \
    src    = convert_float4(src0); \
    tmp.x  = sigmoid_(src.x * beta, logE); \
    data.x = src.x * tmp.x; \
    int4 dst = convert_int4(data); \
    write_imagei(output, coord, dst);

__kernel void swish_I32toI32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    SWISH_I32_I32_PROCESS()
}

__kernel void swish_I32toI32_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    SWISH_I32_I32_PROCESS()
}

#define SWISH_F32_U8_PROCESS() \
    float4 src, tmp, data; \
    src = read_imagef(input, coord); \
    tmp.x = sigmoid_(src.x * beta, logE); \
    data.x = src.x * tmp.x; \
    uint4 dst = convert_uint4_rte(data * outputScale + outputZP); \
    write_imageui(output, coord, dst);

__kernel void swish_F32toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    SWISH_F32_U8_PROCESS()
}

__kernel void swish_F32toU8_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP,
                 float            beta,
                 float            logE)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    SWISH_F32_U8_PROCESS()
}