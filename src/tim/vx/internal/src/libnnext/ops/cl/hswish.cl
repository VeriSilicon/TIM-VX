#define HSWISH_F32_F32_PROCESS() \
    float4 src, tmp, dst; \
    src   = read_imagef(input, coord); \
    tmp   = src + 3; \
    tmp   = tmp > 0 ? tmp : 0; \
    tmp   = tmp < 6 ? tmp : 6; \
    dst   = src * tmp / 6.0f; \
    write_imagef(output, coord, dst);

__kernel void hswish_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    HSWISH_F32_F32_PROCESS()
}

__kernel void hswish_F32toF32_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    HSWISH_F32_F32_PROCESS()
}


#define HSWISH_U8_U8_PROCESS() \
    float4 src, tmp, data; \
    uint4 src0 = read_imageui(input, coord); \
    src   = convert_float4(src0) * inputScale - inputTail; \
    tmp   = src + 3; \
    tmp   = tmp > 0 ? tmp : 0; \
    tmp   = tmp < 6 ? tmp : 6; \
    data   = src * tmp / 6.0f; \
    uint4 dst = convert_uint4(data * outputScale + outputZP); \
    write_imageui(output, coord, dst);

__kernel void hswish_U8toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    HSWISH_U8_U8_PROCESS()
}

__kernel void hswish_U8toU8_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    HSWISH_U8_U8_PROCESS()
}


#define HSWISH_I32_I32_PROCESS() \
    int4 tmp, dst, src; \
    src   = read_imagei(input, coord); \
    tmp   = src + 3; \
    tmp   = tmp > 0 ? tmp : 0; \
    tmp   = tmp < 6 ? tmp : 6; \
    dst   = src * tmp / 6; \
    write_imagei(output, coord, dst);

__kernel void hswish_I32toI32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    HSWISH_I32_I32_PROCESS()
}

__kernel void hswish_I32toI32_2D(
    __read_only  image2d_t        input,
    __write_only image2d_t        output,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputZP)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    HSWISH_I32_I32_PROCESS()
}
