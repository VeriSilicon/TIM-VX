float evaluate_polynomial_alpha(float x2)
{
    float4 alpha0 = (float4){-2.72614225801306e-10f, 2.77068142495902e-08f,
                            -2.10102402082508e-06f, -5.69250639462346e-05f};
    float4 alpha1 = (float4){-7.34990630326855e-04f, -2.95459980854025e-03f,
                            -1.60960333262415e-02f, 0};

    float poly = alpha0.x * x2 + alpha0.y;
    poly = poly * x2 + alpha0.z;
    poly = poly * x2 + alpha0.w;
    poly = poly * x2 + alpha1.x;
    poly = poly * x2 + alpha1.y;
    poly = poly * x2 + alpha1.z;

    return poly;
}

float evaluate_polynomial_beta(float x2)
{
    float4 beta0 = (float4){-1.45660718464996e-05f, -2.13374055278905e-04f,
                            -1.68282697438203e-03f, -7.37332916720468e-03f};
    float4 beta1 = (float4){-1.42647390514189e-02f, 0, 0, 0};

    float poly = beta0.x * x2 + beta0.y;
    poly = poly * x2 + beta0.z;
    poly = poly * x2 + beta0.w;
    poly = poly * x2 + beta1.x;

    return 1.0f / poly;
}

float eltwise_unary_erf(float _x)
{
    float x = clamp(_x, -4, 4);
    float x2 = x * x;

    return x * evaluate_polynomial_alpha(x2) * evaluate_polynomial_beta(x2);
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
