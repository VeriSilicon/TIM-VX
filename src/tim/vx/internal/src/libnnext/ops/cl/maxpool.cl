#define VSI_FLOAT32_MIN     (1.175494351e-38F)

#define MAXPOOL_QINT(in_name, out_name, src_type, dst_type, max_val, read_func, write_func, conv_func) \
__kernel void maxpool_##in_name##to##out_name( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output, \
                 int              width, \
                 int              height, \
                 int              stride_x, \
                 int              stride_y, \
                 int              pad_x, \
                 int              pad_y, \
                 int              kernel_dia_x, \
                 int              kernel_dia_y, \
                 int              dilation_x, \
                 int              dilation_y, \
                 float            inout_scale, \
                 float            inout_tail) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int gidz = get_global_id(2); \
    int4 coord_out = (int4)(gidx, gidy, gidz, 0); \
    int2 pos_start = coord_out.xy * (int2)(stride_x, stride_y) - (int2)(pad_x, pad_y); \
    int4 coord_in = coord_out; \
    int2 pos_end = pos_start + (int2)(kernel_dia_x, kernel_dia_y); \
 \
    for(; pos_start.x < 0;) \
    { \
        pos_start.x += dilation_x; \
    } \
    for(; pos_start.y < 0;) \
    { \
        pos_start.y += dilation_y; \
    } \
 \
    pos_end = min(pos_end, (int2)(width, height)); \
 \
    src_type src0, maxVal; \
    maxVal.x = max_val; \
 \
    for(coord_in.y = pos_start.y; coord_in.y < pos_end.y; coord_in.y += dilation_y) \
    { \
        for(coord_in.x = pos_start.x; coord_in.x < pos_end.x;) \
        { \
            src0 = read_func(input, coord_in); \
            coord_in.x += dilation_x; \
            maxVal = max(src0, maxVal); \
        } \
    } \
 \
    float4 fValTmp; \
    fValTmp.x = maxVal.x * inout_scale + inout_tail; \
    dst_type dst = conv_func(fValTmp); \
    write_func(output, coord_out, dst.xxxx); \
}
MAXPOOL_QINT(U32, U32, uint4, uint4, 0, read_imageui, write_imageui, convert_uint4_rte)
MAXPOOL_QINT(I32, I32, int4, int4, -2147483648, read_imagei, write_imagei, convert_int4_rte)

__kernel void maxpool_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 int              width,
                 int              height,
                 int              stride_x,
                 int              stride_y,
                 int              pad_x,
                 int              pad_y,
                 int              kernel_dia_x,
                 int              kernel_dia_y,
                 int              dilation_x,
                 int              dilation_y,
                 float            inout_scale,
                 float            inout_tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int gidz = get_global_id(2);
    int4 coord_out = (int4)(gidx, gidy, gidz, 0);
    int2 pos_start = coord_out.xy * (int2)(stride_x, stride_y) - (int2)(pad_x, pad_y);
    int4 coord_in = coord_out;
    int2 pos_end = pos_start + (int2)(kernel_dia_x, kernel_dia_y);

    for(; pos_start.x < 0;)
    {
        pos_start.x += dilation_x;
    }
    for(; pos_start.y < 0;)
    {
        pos_start.y += dilation_y;
    }

    pos_end = min(pos_end, (int2)(width, height));

    float4 src0, maxVal;
    maxVal.x = VSI_FLOAT32_MIN;

    for(coord_in.y = pos_start.y; coord_in.y < pos_end.y; coord_in.y += dilation_y)
    {
        for(coord_in.x = pos_start.x; coord_in.x < pos_end.x;)
        {
            src0 = read_imagef(input, coord_in);
            coord_in.x += dilation_x;
            maxVal = max(src0, maxVal);
        }
    }

    write_imagef(output, coord_out, maxVal.xxxx);
}

__kernel void maxpool_U32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 int              width,
                 int              height,
                 int              stride_x,
                 int              stride_y,
                 int              pad_x,
                 int              pad_y,
                 int              kernel_dia_x,
                 int              kernel_dia_y,
                 int              dilation_x,
                 int              dilation_y,
                 float            inout_scale,
                 float            inout_tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int gidz = get_global_id(2);
    int4 coord_out = (int4)(gidx, gidy, gidz, 0);
    int2 pos_start = coord_out.xy * (int2)(stride_x, stride_y) - (int2)(pad_x, pad_y);
    int4 coord_in = coord_out;
    int2 pos_end = pos_start + (int2)(kernel_dia_x, kernel_dia_y);

    for(; pos_start.x < 0;)
    {
        pos_start.x += dilation_x;
    }
    for(; pos_start.y < 0;)
    {
        pos_start.y += dilation_y;
    }

    pos_end = min(pos_end, (int2)(width, height));

    uint4 src0, maxVal;
    maxVal.x = 0;

    for(coord_in.y = pos_start.y; coord_in.y < pos_end.y; coord_in.y += dilation_y)
    {
        for(coord_in.x = pos_start.x; coord_in.x < pos_end.x;)
        {
            src0 = read_imageui(input, coord_in);
            coord_in.x += dilation_x;
            maxVal = max(src0, maxVal);
        }
    }

    float4 dst;
    dst.x = maxVal.x * inout_scale + inout_tail;

    write_imagef(output, coord_out, dst.xxxx);
}

__kernel void maxpool_F32toU32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                 int              width,
                 int              height,
                 int              stride_x,
                 int              stride_y,
                 int              pad_x,
                 int              pad_y,
                 int              kernel_dia_x,
                 int              kernel_dia_y,
                 int              dilation_x,
                 int              dilation_y,
                 float            inout_scale,
                 float            inout_tail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int gidz = get_global_id(2);
    int4 coord_out = (int4)(gidx, gidy, gidz, 0);
    int2 pos_start = coord_out.xy * (int2)(stride_x, stride_y) - (int2)(pad_x, pad_y);
    int4 coord_in = coord_out;
    int2 pos_end = pos_start + (int2)(kernel_dia_x, kernel_dia_y);

    for(; pos_start.x < 0;)
    {
        pos_start.x += dilation_x;
    }
    for(; pos_start.y < 0;)
    {
        pos_start.y += dilation_y;
    }

    pos_end = min(pos_end, (int2)(width, height));

    float4 src0, maxVal;
    maxVal.x = VSI_FLOAT32_MIN;

    for(coord_in.y = pos_start.y; coord_in.y < pos_end.y; coord_in.y += dilation_y)
    {
        for(coord_in.x = pos_start.x; coord_in.x < pos_end.x;)
        {
            src0 = read_imagef(input, coord_in);
            coord_in.x += dilation_x;
            maxVal = max(src0, maxVal);
        }
    }

    uint4 dst;
    dst.x = convert_uint_rte(maxVal.x * inout_scale + inout_tail);

    write_imageui(output, coord_out, dst.xxxx);
}
