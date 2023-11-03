#pragma OPENCL EXTENSION cl_viv_vx_extension : enable
#include "cl_viv_vx_ext.h"

#define rlogE    (0.693147182f)

float LOG(float x)
{
    x = log2(x);
    return x * rlogE;
}

__kernel void log_softmax_axis1_F32toF32(
    __read_only   image2d_array_t input,
    __write_only  image2d_array_t output,
    int axis, float beta,
    float scale, float scaleOut, float zpOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int height = get_image_height(input);
    int4 coord_in = (int4)(x, 0, z, 0);
    float4 maxValue;
    float4 src, dst = {0.0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = read_imagef(input, coord_in);
    for (coord_in.y = 1; coord_in.y < height; )
    {
        src = read_imagef(input, coord_in);
        coord_in.y++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.f;
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = read_imagef(input, coord_in);
        coord_in.y++;

        sum += exp2((src.x - maxValue.x) * scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = read_imagef(input, coord_in);

        dst.x = (src.x - maxValue.x) * beta - logSum;
        write_imagef(output, coord_in, dst);
        coord_in.y++;
    }
}

__kernel void log_softmax_axis1_F32toF32_2D(
    __read_only   image2d_t input,
    __write_only  image2d_t output,
    int axis, float beta,
    float scale, float scaleOut, float zpOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int height = get_image_height(input);
    int2 coord_in = (int2)(x, 0);
    float4 maxValue;
    float4 src, dst = {0.0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = read_imagef(input, coord_in);
    for (coord_in.y = 1; coord_in.y < height; )
    {
        src = read_imagef(input, coord_in);
        coord_in.y++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.0f;
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = read_imagef(input, coord_in);
        coord_in.y++;

        sum += exp2((src.x - maxValue.x) * scale);
    }

    // Compute result.
    float logSum = 1.0f * LOG(sum);
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = read_imagef(input, coord_in);

        dst.x = (src.x - maxValue.x) * beta - logSum;
        write_imagef(output, coord_in, dst);
        coord_in.y++;
    }
}

__kernel void log_softmax_axis1_U8toU8(
    __read_only  image2d_array_t input,
    __write_only image2d_array_t output,
    int axis, float beta,
    float scale, float scaleOut, float zpOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int height = get_image_height(input);
    int4 coord_in = (int4)(x, 0, z, 0);
    float4 maxValue;
    float4 src;
    uint4 dst = {0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = convert_float4(read_imageui(input, coord_in));
    for (coord_in.y = 1; coord_in.y < height; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.y++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.f;
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.y++;

        sum += exp2((src.x - maxValue.x) * scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = convert_float4(read_imageui(input, coord_in));

        dst.x = convert_uint(((src.x - maxValue.x) * beta - logSum) * scaleOut + zpOut);

        write_imageui(output, coord_in, dst);
        coord_in.y++;
    }
}

__kernel void log_softmax_axis1_U8toU8_2D(
    __read_only  image2d_t input,
    __write_only image2d_t output,
    int axis, float beta,
    float scale, float scaleOut, float zpOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int height = get_image_height(input);
    int2 coord_in = (int2)(x, 0);
    float4 maxValue;
    float4 src;
    uint4 dst = {0};

    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    maxValue = convert_float4(read_imageui(input, coord_in));
    for (coord_in.y = 1; coord_in.y < height; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.y++;

        maxValue = maxValue > src ? maxValue : src;
    }

    // Compute sum.
    float sum = 0.f;
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = convert_float4(read_imageui(input, coord_in));
        coord_in.y++;

        sum += exp2((src.x - maxValue.x)*scale);
    }

    // Compute result.
    float logSum = LOG(sum);
    for (coord_in.y = 0; coord_in.y < height; )
    {
        src = convert_float4(read_imageui(input, coord_in));

        dst.x = convert_uint(((src.x - maxValue.x) * beta - logSum) * scaleOut + zpOut);
        write_imageui(output, coord_in, dst);
        coord_in.y++;
    }
}

__kernel void log_softmax_axis1_BF16oBF16(
    __read_only   image2d_array_t input,
    __write_only  image2d_array_t output,
        int axis, float beta,
        float scale, float scaleOut, float zpOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int height = get_image_height(input);
    int4 coord_in = (int4)(x, 0, z, 0);
    float4 maxValue, src, dst = {0.0};
    uint4 data, val, out;

    data = read_imageui(input, coord_in);
    data = data << 16;
    _viv_asm(COPY, maxValue, data, 16);
    for (coord_in.y = 1; coord_in.y < height; )
    {
        data = read_imageui(input, coord_in);
        coord_in.y++;
        data = data << 16;
        _viv_asm(COPY, src, data, 16);

        maxValue = maxValue > src ? maxValue : src;
    }

    float sum = 0.f;
    for (coord_in.y = 0; coord_in.y < height; )
    {
        data = read_imageui(input, coord_in);
        coord_in.y++;
        data = data << 16;
        _viv_asm(COPY, src, data, 16);

        sum += exp2((src.x - maxValue.x) * scale);
    }

    float logSum = LOG(sum);
    for (coord_in.y = 0; coord_in.y < height; )
    {
        data = read_imageui(input, coord_in);
        data = data << 16;
        _viv_asm(COPY, src, data, 16);

        dst.x = (src.x - maxValue.x) * beta - logSum;

        _viv_asm(COPY, val, dst, 16);
        out = val >> 16;

        write_imageui(output, coord_in, out);
        coord_in.y++;
    }
}

__kernel void log_softmax_axis1_BF16toBF16_2D(
    __read_only   image2d_t input,
    __write_only  image2d_t output,
        int axis, float beta,
        float scale, float scaleOut, float zpOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int height = get_image_height(input);
    int2 coord_in = (int2)(x, 0);
    float4 maxValue, src, dst = {0.0};
    uint4 data, val, out;

    data = read_imageui(input, coord_in);
    data = data << 16;
    _viv_asm(COPY, maxValue, data, 16);
    for (coord_in.y = 1; coord_in.y < height; )
    {
        data = read_imageui(input, coord_in);
        coord_in.y++;
        data = data << 16;
        _viv_asm(COPY, src, data, 16);

        maxValue = maxValue > src ? maxValue : src;
    }

    float sum = 0.0f;
    for (coord_in.y = 0; coord_in.y < height; )
    {
        data = read_imageui(input, coord_in);
        coord_in.y++;
        data = data << 16;
        _viv_asm(COPY, src, data, 16);

        sum += exp2((src.x - maxValue.x) * scale);
    }

    float logSum = 1.0f * LOG(sum);
    for (coord_in.y = 0; coord_in.y < height; )
    {
        data = read_imageui(input, coord_in);
        data = data << 16;
        _viv_asm(COPY, src, data, 16);

        dst.x = (src.x - maxValue.x) * beta - logSum;
        _viv_asm(COPY, val, dst, 16);
        out = val >> 16;
        write_imageui(output, coord_in, out);
        coord_in.y++;
    }
}
#undef rlogE
