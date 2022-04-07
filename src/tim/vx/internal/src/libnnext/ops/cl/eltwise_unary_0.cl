float eltwise_unary_sin(float x, float alpha, float beta)
{
    return native_sin(x);
}

float eltwise_unary_cos(float x, float alpha, float beta)
{
    return native_cos(x);
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

float erf_eval(float _x)
{
    float x = clamp(_x, -4, 4);
    float x2 = x * x;

    return x * evaluate_polynomial_alpha(x2) * evaluate_polynomial_beta(x2);
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

float eltwise_unary_selu(float val, float alpha_times_gamma, float gamma)
{
    float x = val * logE;
    x = exp2(x) * alpha_times_gamma - alpha_times_gamma;

    return val <= 0 ? x : val * gamma;
}

float eltwise_unary_celu(float val, float alpha, float rcp_alpha)
{
    float x = val * logE * rcp_alpha;
    x = exp2(x) * alpha - alpha;

    return val < 0 ? x : val;
}

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
ELTWISE_UNARY_F32_2D(cos)
ELTWISE_UNARY_F32_2D(exp)
ELTWISE_UNARY_F32_2D(log)
ELTWISE_UNARY_F32_2D(neg)
ELTWISE_UNARY_F32_2D(mish)
ELTWISE_UNARY_F32_2D(hard_sigmoid)
ELTWISE_UNARY_F32_2D(round)
ELTWISE_UNARY_F32_2D(gelu)
ELTWISE_UNARY_F32_2D(hard_gelu)
ELTWISE_UNARY_F32_2D(selu)
ELTWISE_UNARY_F32_2D(celu)

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
ELTWISE_UNARY_U8_2D(cos)
ELTWISE_UNARY_U8_2D(exp)
ELTWISE_UNARY_U8_2D(log)
ELTWISE_UNARY_U8_2D(neg)
ELTWISE_UNARY_U8_2D(mish)
ELTWISE_UNARY_U8_2D(hard_sigmoid)
ELTWISE_UNARY_U8_2D(round)
ELTWISE_UNARY_U8_2D(gelu)
ELTWISE_UNARY_U8_2D(hard_gelu)
ELTWISE_UNARY_U8_2D(selu)
ELTWISE_UNARY_U8_2D(celu)

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
