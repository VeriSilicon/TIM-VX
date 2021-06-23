#define MUL2_RSQRTPI    (1.1283791670955126f)
float eltwise_unary_erf(float x)
{
    float res = 0;
    float tmp = x;
    float factorial = 1;
    float x_pow = x;
    float one = 1.0f;
    float n = 1;

    while (fabs(tmp) > 1e-5)
    {
        res += tmp;

        factorial *= n;
        one *= -1;
        x_pow *= x * x;
        tmp = one / factorial * x_pow / ( 2 * n + 1);

        n += 1.0f;
    }
    return res * MUL2_RSQRTPI;
}

#define ELTWISE_UNARY_F32(func_name) \
__kernel void func_name##_F32toF32 \
    ( \
    __read_only  image2d_array_t input, \
    __write_only image2d_array_t output, \
                 float           inputScale, \
                 float           inputTail, \
                 float           outputScale, \
                 float           outputZP \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    float4 src = read_imagef(input, coord); \
 \
    float4 dst = 0; \
    dst.x = eltwise_unary_##func_name(src.x); \
 \
    write_imagef(output, coord, dst); \
}
ELTWISE_UNARY_F32(erf)

#define ELTWISE_UNARY_F32_2D(func_name) \
__kernel void func_name##_F32toF32_2D \
    ( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
                 float     inputScale, \
                 float     inputTail, \
                 float     outputScale, \
                 float     outputZP \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src = read_imagef(input, coord); \
 \
    float4 dst = 0; \
    dst.x = eltwise_unary_##func_name(src.x); \
 \
    write_imagef(output, coord, dst); \
}
ELTWISE_UNARY_F32_2D(erf)

#define ELTWISE_UNARY_U8(func_name) \
__kernel void func_name##_U8toU8 \
    ( \
    __read_only  image2d_array_t input, \
    __write_only image2d_array_t output, \
                 float           inputScale, \
                 float           inputTail, \
                 float           outputScale, \
                 float           outputZP \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    uint4 src = read_imageui(input, coord); \
    float4 data = convert_float4(src) * inputScale - inputTail; \
 \
    data.x = eltwise_unary_##func_name(data.x); \
    uint4 dst = convert_uint4(data * outputScale + outputZP); \
 \
    write_imageui(output, coord, dst); \
}
ELTWISE_UNARY_U8(erf)

#define ELTWISE_UNARY_U8_2D(func_name) \
__kernel void func_name##_U8toU8_2D \
    ( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
                 float     inputScale, \
                 float     inputTail, \
                 float     outputScale, \
                 float     outputZP \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    uint4 src = read_imageui(input, coord); \
    float4 data = convert_float4(src) * inputScale - inputTail; \
 \
    data.x = eltwise_unary_##func_name(data.x); \
    uint4 dst = convert_uint4(data * outputScale + outputZP); \
 \
    write_imageui(output, coord, dst); \
}
ELTWISE_UNARY_U8_2D(erf)
