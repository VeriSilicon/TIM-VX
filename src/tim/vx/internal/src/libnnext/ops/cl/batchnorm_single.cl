
__kernel void batch_norm_F32toF32
    (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t Mean,
    __read_only  image2d_array_t Variance,
    __read_only  image2d_array_t Gamma,
    __read_only  image2d_array_t Beta,
    __write_only image2d_array_t output,
                 float           eps,
                 float           input_scale,
                 float           input_tail,
                 float           output_scale,
                 float           output_zp
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    float4 src, mean, var, gamma, beta;
    readImage2DArray(src, input, coord);
    readImage2DArray(mean, Mean, coord);
    readImage2DArray(var, Variance, coord);
    readImage2DArray(gamma, Gamma, coord);
    readImage2DArray(beta, Beta, coord);

    float4 dst;
    src.x = src.x - mean.x;
    float inv = rsqrt(var.x + eps);
    dst.x = src.x * inv *gamma.x + beta.x;

    write_imagef(output, coord, dst);
}

__kernel void batch_norm_F32toF32_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t Mean,
    __read_only  image2d_t Variance,
    __read_only  image2d_t Gamma,
    __read_only  image2d_t Beta,
    __write_only image2d_t output,
                 float     eps,
                 float     input_scale,
                 float     input_tail,
                 float     output_scale,
                 float     output_zp
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    float4 src = read_imagef(input, coord);
    float4 mean = read_imagef(Mean, coord);
    float4 var = read_imagef(Variance, coord);
    float4 gamma = read_imagef(Gamma, coord);
    float4 beta = read_imagef(Beta, coord);

    float4 dst = 0;
    src.x = src.x - mean.x;
    float inv = rsqrt(var.x + eps);
    dst.x = src.x * inv *gamma.x + beta.x;

    write_imagef(output, coord, dst);
}

__kernel void batch_norm_U8toU8
    (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t Mean,
    __read_only  image2d_array_t Variance,
    __read_only  image2d_array_t Gamma,
    __read_only  image2d_array_t Beta,
    __write_only image2d_array_t output,
                 float           eps,
                 float           input_scale,
                 float           input_tail,
                 float           output_scale,
                 float           output_zp
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    uint4 data;
    float4 src, mean, var, gamma, beta;
    readImage2DArray(data, input, coord);
    readImage2DArray(mean, Mean, coord);
    readImage2DArray(var, Variance, coord);
    readImage2DArray(gamma, Gamma, coord);
    readImage2DArray(beta, Beta, coord);

    src = convert_float4(data) * input_scale - input_tail;
    src.x = src.x - mean.x;
    float inv = rsqrt(var.x + eps);
    src.x = src.x * inv *gamma.x + beta.x;

    uint4 dst = convert_uint4(src * output_scale + output_zp);

    write_imageui(output, coord, dst);
}

__kernel void batch_norm_U8toU8_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t Mean,
    __read_only  image2d_t Variance,
    __read_only  image2d_t Gamma,
    __read_only  image2d_t Beta,
    __write_only image2d_t output,
                 float     eps,
                 float     input_scale,
                 float     input_tail,
                 float     output_scale,
                 float     output_zp
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    uint4  data = read_imageui(input, coord);
    float4 mean = read_imagef(Mean, coord);
    float4 var = read_imagef(Variance, coord);
    float4 gamma = read_imagef(Gamma, coord);
    float4 beta = read_imagef(Beta, coord);

    float4 src = convert_float4(data) * input_scale - input_tail;
    src.x = src.x - mean.x;
    float inv = rsqrt(var.x + eps);
    src.x = src.x * inv *gamma.x + beta.x;

    uint4 dst = convert_uint4(src * output_scale + output_zp);

    write_imageui(output, coord, dst);
}

