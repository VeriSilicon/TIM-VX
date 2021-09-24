#define BN_U8_SAVE \
    uint4 dst = convert_uint4(src * output_scale + output_zp); \
    write_imageui(output, coord, dst);

#define BN_I32_SAVE \
    int4 dst = convert_int4(src * output_scale + output_zp); \
    write_imagei(output, coord, dst);

#define BN_F32_SAVE \
    write_imagef(output, coord, src);

#define BATCH_NORM_F32_SH_IMPL(TYPE) \
__kernel void batch_norm_F32to##TYPE \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_array_t Mean, \
    __read_only  image2d_array_t Variance, \
    __read_only  image2d_array_t Gamma, \
    __read_only  image2d_array_t Beta, \
    __write_only image2d_array_t output, \
                 float           eps, \
                 float           input_scale, \
                 float           input_tail, \
                 float           output_scale, \
                 float           output_zp \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    float4 src, mean, var, gamma, beta; \
    READ_IMAGEF_2DARRAY(src, input, coord); \
    READ_IMAGEF_2DARRAY(mean, Mean, coord); \
    READ_IMAGEF_2DARRAY(var, Variance, coord); \
    READ_IMAGEF_2DARRAY(gamma, Gamma, coord); \
    READ_IMAGEF_2DARRAY(beta, Beta, coord); \
 \
    src.x = src.x - mean.x; \
    float inv = rsqrt(var.x + eps); \
    src.x = src.x * inv *gamma.x + beta.x; \
 \
    BN_##TYPE##_SAVE \
}
BATCH_NORM_F32_SH_IMPL(F32)
BATCH_NORM_F32_SH_IMPL(U8)
BATCH_NORM_F32_SH_IMPL(I32)

#define BATCH_NORM_F32_SH_IMPL_2D(TYPE) \
__kernel void batch_norm_F32to##TYPE##_2D \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t Mean, \
    __read_only  image2d_t Variance, \
    __read_only  image2d_t Gamma, \
    __read_only  image2d_t Beta, \
    __write_only image2d_t output, \
                 float     eps, \
                 float     input_scale, \
                 float     input_tail, \
                 float     output_scale, \
                 float     output_zp \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    float4 src = read_imagef(input, coord); \
    float4 mean = read_imagef(Mean, coord); \
    float4 var = read_imagef(Variance, coord); \
    float4 gamma = read_imagef(Gamma, coord); \
    float4 beta = read_imagef(Beta, coord); \
 \
    src.x = src.x - mean.x; \
    float inv = rsqrt(var.x + eps); \
    src.x = src.x * inv *gamma.x + beta.x; \
 \
    BN_##TYPE##_SAVE \
}
BATCH_NORM_F32_SH_IMPL_2D(F32)
BATCH_NORM_F32_SH_IMPL_2D(U8)
BATCH_NORM_F32_SH_IMPL_2D(I32)

#define BATCH_NORM_U8_SH_IMPL(TYPE) \
__kernel void batch_norm_U8to##TYPE \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_array_t Mean, \
    __read_only  image2d_array_t Variance, \
    __read_only  image2d_array_t Gamma, \
    __read_only  image2d_array_t Beta, \
    __write_only image2d_array_t output, \
                 float           eps, \
                 float           input_scale, \
                 float           input_tail, \
                 float           output_scale, \
                 float           output_zp \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    uint4 data; \
    float4 src, mean, var, gamma, beta; \
    READ_IMAGEUI_2DARRAY(data, input, coord); \
    READ_IMAGEF_2DARRAY(mean, Mean, coord); \
    READ_IMAGEF_2DARRAY(var, Variance, coord); \
    READ_IMAGEF_2DARRAY(gamma, Gamma, coord); \
    READ_IMAGEF_2DARRAY(beta, Beta, coord); \
 \
    src = convert_float4(data) * input_scale - input_tail; \
    src.x = src.x - mean.x; \
    float inv = rsqrt(var.x + eps); \
    src.x = src.x * inv *gamma.x + beta.x; \
 \
    BN_##TYPE##_SAVE \
}
BATCH_NORM_U8_SH_IMPL(U8)
BATCH_NORM_U8_SH_IMPL(F32)

#define BATCH_NORM_U8_SH_IMPL_2D(TYPE) \
__kernel void batch_norm_U8to##TYPE##_2D \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t Mean, \
    __read_only  image2d_t Variance, \
    __read_only  image2d_t Gamma, \
    __read_only  image2d_t Beta, \
    __write_only image2d_t output, \
                 float     eps, \
                 float     input_scale, \
                 float     input_tail, \
                 float     output_scale, \
                 float     output_zp \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    uint4  data = read_imageui(input, coord); \
    float4 mean = read_imagef(Mean, coord); \
    float4 var = read_imagef(Variance, coord); \
    float4 gamma = read_imagef(Gamma, coord); \
    float4 beta = read_imagef(Beta, coord); \
 \
    float4 src = convert_float4(data) * input_scale - input_tail; \
    src.x = src.x - mean.x; \
    float inv = rsqrt(var.x + eps); \
    src.x = src.x * inv *gamma.x + beta.x; \
 \
    BN_##TYPE##_SAVE \
}
BATCH_NORM_U8_SH_IMPL_2D(U8)
BATCH_NORM_U8_SH_IMPL_2D(F32)

#define BATCH_NORM_I32_SH_IMPL(TYPE) \
__kernel void batch_norm_I32to##TYPE \
    ( \
    __read_only  image2d_array_t input, \
    __read_only  image2d_array_t Mean, \
    __read_only  image2d_array_t Variance, \
    __read_only  image2d_array_t Gamma, \
    __read_only  image2d_array_t Beta, \
    __write_only image2d_array_t output, \
                 float           eps, \
                 float           input_scale, \
                 float           input_tail, \
                 float           output_scale, \
                 float           output_zp \
    ) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    int4 data; \
    float4 src, mean, var, gamma, beta; \
    READ_IMAGEI_2DARRAY(data, input, coord); \
    READ_IMAGEF_2DARRAY(mean, Mean, coord); \
    READ_IMAGEF_2DARRAY(var, Variance, coord); \
    READ_IMAGEF_2DARRAY(gamma, Gamma, coord); \
    READ_IMAGEF_2DARRAY(beta, Beta, coord); \
 \
    src = convert_float4(data) * input_scale - input_tail; \
    src.x = src.x - mean.x; \
    float inv = rsqrt(var.x + eps); \
    src.x = src.x * inv *gamma.x + beta.x; \
 \
    BN_##TYPE##_SAVE \
}
BATCH_NORM_I32_SH_IMPL(I32)
BATCH_NORM_I32_SH_IMPL(F32)

#define BATCH_NORM_I32_SH_IMPL_2D(TYPE) \
__kernel void batch_norm_I32to##TYPE##_2D \
    ( \
    __read_only  image2d_t input, \
    __read_only  image2d_t Mean, \
    __read_only  image2d_t Variance, \
    __read_only  image2d_t Gamma, \
    __read_only  image2d_t Beta, \
    __write_only image2d_t output, \
                 float     eps, \
                 float     input_scale, \
                 float     input_tail, \
                 float     output_scale, \
                 float     output_zp \
    ) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
 \
    int4  data = read_imagei(input, coord); \
    float4 mean = read_imagef(Mean, coord); \
    float4 var = read_imagef(Variance, coord); \
    float4 gamma = read_imagef(Gamma, coord); \
    float4 beta = read_imagef(Beta, coord); \
 \
    float4 src = convert_float4(data) * input_scale - input_tail; \
    src.x = src.x - mean.x; \
    float inv = rsqrt(var.x + eps); \
    src.x = src.x * inv *gamma.x + beta.x; \
 \
    BN_##TYPE##_SAVE \
}
BATCH_NORM_I32_SH_IMPL_2D(I32)
BATCH_NORM_I32_SH_IMPL_2D(F32)