#define FP32_MIN   -3.4e38
#define I32_MIN    -2147483647

__kernel void maxpoolwithargmax_F32toF32_I32_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    __write_only image2d_t  argmax,
    int ksize_x, int ksize_y, int stride_x, int stride_y,
    int pad_left, int pad_top, int width, int height,
    float scale, float tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int2 coord_out = (int2)(gidx, gidy);
    int2 coord_in  = coord_out;

    int hstart = gidy * stride_y - pad_top;
    int wstart = gidx * stride_x - pad_left;
    int hend = min(hstart + ksize_y, height);
    int wend = min(wstart + ksize_x, width);
    int h, w;
    int4 index_max = (int4)(0);
    float value_max = FP32_MIN;
    float4 dst = (float4)(0);

    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int2 coord_max = (int2)(wstart, hstart);
    for (h = hstart; h < hend; ++ h)
    {
        for (w = wstart; w < wend; ++ w)
        {
            coord_in.xy = (int2)(w, h);
            float4 data = read_imagef(input, coord_in);

            if (data.x > value_max)
            {
                value_max = data.x;
                coord_max = coord_in;
            }
        }
    }

    index_max.x = coord_max.x + coord_max.y * width;
    dst.x = value_max;
    write_imagef(output, coord_out, dst);
    write_imagei(argmax, coord_out, index_max);
}

__kernel void maxpoolwithargmax_BF16toBF16_I32_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    __write_only image2d_t  argmax,
    int ksize_x, int ksize_y, int stride_x, int stride_y,
    int pad_left, int pad_top, int width, int height,
    float scale, float tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int2 coord_out = (int2)(gidx, gidy);
    int2 coord_in  = coord_out;

    int hstart = gidy * stride_y - pad_top;
    int wstart = gidx * stride_x - pad_left;
    int hend = min(hstart + ksize_y, height);
    int wend = min(wstart + ksize_x, width);
    int h, w;
    int4 index_max = (int4)(0);
    float value_max = FP32_MIN;
    uint4 dst = (uint4)(0);

    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int2 coord_max = (int2)(wstart, hstart);
    for (h = hstart; h < hend; ++ h)
    {
        for (w = wstart; w < wend; ++ w)
        {
            coord_in.xy = (int2)(w, h);
            uint4 src = read_imageui(input, coord_in);
            src = src << 16;
            float4 data;
            _viv_asm(COPY, data, src, 16);

            if (data.x > value_max)
            {
                value_max = data.x;
                coord_max = coord_in;
            }
        }
    }

    index_max.x = coord_max.x + coord_max.y * width;
    _viv_asm(COPY, dst, value_max, 4);
    dst.x = dst.x >> 16;
    write_imageui(output, coord_out, dst);
    write_imagei(argmax, coord_out, index_max);
}

__kernel void maxpoolwithargmax_U32toU32_I32_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    __write_only image2d_t  argmax,
    int ksize_x, int ksize_y, int stride_x, int stride_y,
    int pad_left, int pad_top, int width, int height,
    float scale, float tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int2 coord_out = (int2)(gidx, gidy);
    int2 coord_in  = coord_out;

    int hstart = gidy * stride_y - pad_top;
    int wstart = gidx * stride_x - pad_left;
    int hend = min(hstart + ksize_y, height);
    int wend = min(wstart + ksize_x, width);
    int h, w;
    int4 index_max = (int4)(0);
    uint value_max = 0;
    uint4 dst = (uint4)(0);

    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int2 coord_max = (int2)(wstart, hstart);
    for (h = hstart; h < hend; ++ h)
    {
        for (w = wstart; w < wend; ++ w)
        {
            coord_in.xy = (int2)(w, h);
            uint4 data = read_imageui(input, coord_in);

            if (data.x > value_max)
            {
                value_max = data.x;
                coord_max = coord_in;
            }
        }
    }

    index_max.x = coord_max.x + coord_max.y * width;
    dst.x = convert_uint(convert_float(value_max) * scale + tail);
    write_imageui(output, coord_out, dst);
    write_imagei(argmax, coord_out, index_max);
}

__kernel void maxpoolwithargmax_I32toI32_I32_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    __write_only image2d_t  argmax,
    int ksize_x, int ksize_y, int stride_x, int stride_y,
    int pad_left, int pad_top, int width, int height,
    float scale, float tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int2 coord_out = (int2)(gidx, gidy);
    int2 coord_in  = coord_out;

    int hstart = gidy * stride_y - pad_top;
    int wstart = gidx * stride_x - pad_left;
    int hend = min(hstart + ksize_y, height);
    int wend = min(wstart + ksize_x, width);
    int h, w;
    int4 index_max = (int4)(0);
    int value_max = I32_MIN;
    int4 dst = (int4)(0);

    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int2 coord_max = (int2)(wstart, hstart);
    for (h = hstart; h < hend; ++ h)
    {
        for (w = wstart; w < wend; ++ w)
        {
            coord_in.xy = (int2)(w, h);
            int4 data = read_imagei(input, coord_in);

            if (data.x > value_max)
            {
                value_max = data.x;
                coord_max = coord_in;
            }
        }
    }

    index_max.x = coord_max.x + coord_max.y * width;
    dst.x = convert_int(convert_float(value_max) * scale + tail);
    write_imagei(output, coord_out, dst);
    write_imagei(argmax, coord_out, index_max);
}
