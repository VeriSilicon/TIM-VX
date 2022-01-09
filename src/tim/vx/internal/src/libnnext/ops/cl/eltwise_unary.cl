float eltwise_unary_sin(float x, float alpha, float beta)
{
    return native_sin(x);
}

#define logE        (1.44269502f)
#define twoLogE     (logE * 2.0f)
float eltwise_unary_exp(float x, float alpha, float beta)
{
    x *= logE;
    x = exp2(x);
    return x;
}

#define rlogE    (0.693147182f)
float eltwise_unary_log(float x, float alpha, float beta)
{
    x = log2(x);
    return x * rlogE;
}

float eltwise_unary_elu(float val, float alpha, float beta)
{
    float x = val * logE;
    x = exp2(x) * alpha - alpha;

    return val < 0 ? x : val;
}

float eltwise_unary_neg(float x, float alpha, float beta)
{
    return x * -1;
}

float eltwise_unary_hard_sigmoid(float x, float alpha, float beta)
{
    x = alpha * x + beta;
    x = clamp(x, 0, 1);
    return x;
}

float _softrelu(float x, float alpha)
{
    x *= logE;
    x = exp2(x);
    x += 1;
    x = log2(x);
    return x * rlogE;
}

float _tanh(float x, float alpha)
{
    x *= -twoLogE;
    x = 1 + exp2(x);
    x = 1 / x;
    return (2 * x - 1);
}

float eltwise_unary_mish(float x, float alpha, float beta)
{
    float y = _softrelu(x, alpha);
    x = x * _tanh(y, alpha);
    return x;
}

float eltwise_unary_round(float x, float alpha, float beta)
{
    return convert_float(convert_int_rte(x));
}

#define MUL2_RSQRTPI    (1.1283791670955126f)
float erf_eval(float x)
{
    float res = 0;
    float tmp = x;
    float factorial = 1;
    float x_pow = x;
    float one = 1.0f;
    float n = 1;

    if (x <= -3)
        return -1;
    else if (x >= 3)
        return 1;

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
#define RSQRT2      (0.70710678118654752440084436210485f)
float eltwise_unary_gelu(float x, float alpha, float beta)
{
    x = 0.5f * x * (1 + erf_eval(x * RSQRT2));

    return x;
}

#define SQRT_2_RCP_PI  0.7978845834732056f
float eltwise_unary_hard_gelu(float x, float alpha, float beta)
{
    float cdf = 0.5f + 0.5f * _tanh(SQRT_2_RCP_PI *
                        (x + 0.044715f * x * x * x), 0);
    return x * cdf;
}

#define ELTWISE_UNARY_F32(func_name) \
__kernel void func_name##_F32toF32 \
    ( \
    __read_only  image2d_array_t input, \
    __write_only image2d_array_t output, \
                 float           inputScale, \
                 float           inputTail, \
                 float           outputScale, \
                 float           outputZP, \
                 float           alpha, \
                 float           beta \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    float4 src = read_imagef(input, coord); \
 \
    float4 dst = 0; \
    dst.x = eltwise_unary_##func_name(src.x, alpha, beta); \
 \
    write_imagef(output, coord, dst.xxxx); \
}
ELTWISE_UNARY_F32(sin)
ELTWISE_UNARY_F32(exp)
ELTWISE_UNARY_F32(log)
ELTWISE_UNARY_F32(elu)
ELTWISE_UNARY_F32(neg)
ELTWISE_UNARY_F32(mish)
ELTWISE_UNARY_F32(hard_sigmoid)
ELTWISE_UNARY_F32(round)
ELTWISE_UNARY_F32(gelu)
ELTWISE_UNARY_F32(hard_gelu)

#define ELTWISE_UNARY_F32_2D(func_name) \
__kernel void func_name##_F32toF32_2D \
    ( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
                 float     inputScale, \
                 float     inputTail, \
                 float     outputScale, \
                 float     outputZP, \
                 float     alpha, \
                 float           beta \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src = read_imagef(input, coord); \
 \
    float4 dst = 0; \
    dst.x = eltwise_unary_##func_name(src.x, alpha, beta); \
 \
    write_imagef(output, coord, dst.xxxx); \
}
ELTWISE_UNARY_F32_2D(sin)
ELTWISE_UNARY_F32_2D(exp)
ELTWISE_UNARY_F32_2D(log)
ELTWISE_UNARY_F32_2D(elu)
ELTWISE_UNARY_F32_2D(neg)
ELTWISE_UNARY_F32_2D(mish)
ELTWISE_UNARY_F32_2D(hard_sigmoid)
ELTWISE_UNARY_F32_2D(round)
ELTWISE_UNARY_F32_2D(gelu)
ELTWISE_UNARY_F32_2D(hard_gelu)

#define ELTWISE_UNARY_U8(func_name) \
__kernel void func_name##_U8toU8 \
    ( \
    __read_only  image2d_array_t input, \
    __write_only image2d_array_t output, \
                 float           inputScale, \
                 float           inputTail, \
                 float           outputScale, \
                 float           outputZP, \
                 float           alpha, \
                 float           beta \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    uint4 src = read_imageui(input, coord); \
    float4 data = convert_float4(src) * inputScale - inputTail; \
 \
    data.x = eltwise_unary_##func_name(data.x, alpha, beta); \
    uint4 dst = convert_uint4(data * outputScale + outputZP); \
 \
    write_imageui(output, coord, dst); \
}
ELTWISE_UNARY_U8(sin)
ELTWISE_UNARY_U8(exp)
ELTWISE_UNARY_U8(log)
ELTWISE_UNARY_U8(elu)
ELTWISE_UNARY_U8(neg)
ELTWISE_UNARY_U8(mish)
ELTWISE_UNARY_U8(hard_sigmoid)
ELTWISE_UNARY_U8(round)
ELTWISE_UNARY_U8(gelu)
ELTWISE_UNARY_U8(hard_gelu)

#define ELTWISE_UNARY_U8_2D(func_name) \
__kernel void func_name##_U8toU8_2D \
    ( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
                 float     inputScale, \
                 float     inputTail, \
                 float     outputScale, \
                 float     outputZP, \
                 float     alpha, \
                 float     beta \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    uint4 src = read_imageui(input, coord); \
    float4 data = convert_float4(src) * inputScale - inputTail; \
 \
    data.x = eltwise_unary_##func_name(data.x, alpha, beta); \
    uint4 dst = convert_uint4(data * outputScale + outputZP); \
 \
    write_imageui(output, coord, dst); \
}
ELTWISE_UNARY_U8_2D(sin)
ELTWISE_UNARY_U8_2D(exp)
ELTWISE_UNARY_U8_2D(log)
ELTWISE_UNARY_U8_2D(elu)
ELTWISE_UNARY_U8_2D(neg)
ELTWISE_UNARY_U8_2D(mish)
ELTWISE_UNARY_U8_2D(hard_sigmoid)
ELTWISE_UNARY_U8_2D(round)
ELTWISE_UNARY_U8_2D(gelu)
ELTWISE_UNARY_U8_2D(hard_gelu)

__kernel void neg_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_array_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP,
                 float           alpha,
                 float           beta
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 src = read_imagei(input, coord);

    int4 dst = -src;

    write_imagei(output, coord, dst);
}

__kernel void neg_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail,
                 float     outputScale,
                 float     outputZP,
                 float     alpha,
                 float     beta
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int4 src = read_imagei(input, coord);

    int4 dst = -src;

    write_imagei(output, coord, dst);
}
