#define rlogE    (0.693147182f)
float LOG(float x)
{
    x = log2(x);
    return x * rlogE;
}

__kernel void log_softmax_axis0_F32toF32
    (
    __read_only   image2d_array_t input,
    __write_only  image2d_array_t output,
                            int   axis,
                            float beta,
                            float scale,
                            float scaleOut,
                            float zpOut
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int width = get_image_width(input);
    int4 coord_in = (int4)(0, y, z, 0);
    float4 maxValue;
    float4 src, dst = {0.0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = read_imagef(input, coord_in);
    for (coord_in.x = 1; coord_in.x < width; )
    {
        src = read_imagef(input, coord_in);
        coord_in.x++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.f;
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = read_imagef(input, coord_in);
        coord_in.x++;

        sum += exp2((src.x - maxValue.x) * scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = read_imagef(input, coord_in);

        dst.x = (src.x - maxValue.x) * beta - logSum;
        write_imagef(output, coord_in, dst);
        coord_in.x++;
    }
}

__kernel void log_softmax_axis0_F32toF32_2D
    (
    __read_only   image2d_t input,
    __write_only  image2d_t output,
                      int   axis,
                      float beta,
                      float scale,
                      float scaleOut,
                      float zpOut
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_image_width(input);
    int2 coord_in = (int2)(0, y);
    float4 maxValue;
    float4 src, dst = {0.0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = read_imagef(input, coord_in);
    for (coord_in.x = 1; coord_in.x < width; )
    {
        src = read_imagef(input, coord_in);
        coord_in.x++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.0f;
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = read_imagef(input, coord_in);
        coord_in.x++;

        sum += exp2((src.x - maxValue.x) * scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = read_imagef(input, coord_in);

        dst.x = (src.x - maxValue.x) * beta - logSum;
        write_imagef(output, coord_in, dst);
        coord_in.x++;
    }
}

__kernel void log_softmax_axis0_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_array_t output,
                           int   axis,
                           float beta,
                           float scale,
                           float scaleOut,
                           float zpOut
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int width = get_image_width(input);
    int4 coord_in = (int4)(0, y, z, 0);
    float4 maxValue;
    float4 src;
    uint4 dst = {0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = convert_float4(read_imageui(input, coord_in));
    for (coord_in.x = 1; coord_in.x < width; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.x++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.f;
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.x++;

        sum += exp2((src.x - maxValue.x) * scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = convert_float4(read_imageui(input, coord_in));

        dst.x = convert_uint(((src.x - maxValue.x) * beta - logSum) * scaleOut + zpOut);

        write_imageui(output, coord_in, dst);
        coord_in.x++;
    }
}

__kernel void log_softmax_axis0_U8toU8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                     int   axis,
                     float beta,
                     float scale,
                     float scaleOut,
                     float zpOut
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_image_width(input);
    int2 coord_in = (int2)(0, y);
    float4 maxValue;
    float4 src;
    uint4 dst = {0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = convert_float4(read_imageui(input, coord_in));
    for (coord_in.x = 1; coord_in.x < width; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.x++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.f;
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.x++;

        sum += exp2((src.x - maxValue.x)*scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.x = 0; coord_in.x < width; )
    {
        src = convert_float4(read_imageui(input, coord_in));

        dst.x = convert_uint(((src.x - maxValue.x) * beta - logSum) * scaleOut + zpOut);
        write_imageui(output, coord_in, dst);
        coord_in.x++;
    }
}
#undef rlogE
