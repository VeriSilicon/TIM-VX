__kernel void repeat_I32_axis0(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
    int width, int height, int channel, int axis)
{
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(2), 0);
    int4 coord_out = coord;

    for(coord.y = 0; coord.y < height;)
    {
        int4 data = read_imagei(input0, coord);
        int4 len = read_imagei(input1, coord.yw);
        coord.y++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagei(output, coord_out, data);
            coord_out.y++;
        }
    }
}

__kernel void repeat_I32_axis1(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
    int width, int height, int channel, int axis)
{
    int4 coord = (int4)(0, get_global_id(1), get_global_id(2), 0);
    int4 coord_out = coord;

    for(coord.x = 0; coord.x < width;)
    {
        int4 data = read_imagei(input0, coord);
        int4 len = read_imagei(input1, coord.xw);
        coord.x++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagei(output, coord_out, data);
            coord_out.x++;
        }
    }
}

__kernel void repeat_I32_axis2(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
    int width, int height, int channel, int axis)
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int4 coord_out = coord;

    for(coord.z = 0; coord.z < channel;)
    {
        int4 data = read_imagei(input0, coord);
        int4 len = read_imagei(input1, coord.zw);
        coord.z++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagei(output, coord_out, data);
            coord_out.z++;
        }
    }
}

__kernel void repeat_I32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width, int height, int channel, int axis)
{
    int2 coord = (int2)(0, 0);
    int2 coord_out = coord;

    for(coord.x = 0; coord.x < width;)
    {
        int4 data = read_imagei(input0, coord);
        int4 len = read_imagei(input1, coord.xy);
        coord.x++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagei(output, coord_out, data);
            coord_out.x++;
        }
    }
}

__kernel void repeat_F32_axis0(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
    int width, int height, int channel, int axis)
{
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(2), 0);
    int4 coord_out = coord;

    for(coord.y = 0; coord.y < height;)
    {
        float4 data = read_imagef(input0, coord);
        int4 len = read_imagei(input1, coord.yw);
        coord.y++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagef(output, coord_out, data);
            coord_out.y++;
        }
    }
}

__kernel void repeat_F32_axis1(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
    int width, int height, int channel, int axis)
{
    int4 coord = (int4)(0, get_global_id(1), get_global_id(2), 0);
    int4 coord_out = coord;

    for(coord.x = 0; coord.x < width;)
    {
        float4 data = read_imagef(input0, coord);
        int4 len = read_imagei(input1, coord.xw);
        coord.x++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagef(output, coord_out, data);
            coord_out.x++;
        }
    }
}

__kernel void repeat_F32_axis2(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
    int width, int height, int channel, int axis)
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int4 coord_out = coord;

    for(coord.z = 0; coord.z < channel;)
    {
        float4 data = read_imagef(input0, coord);
        int4 len = read_imagei(input1, coord.zw);
        coord.z++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagef(output, coord_out, data);
            coord_out.z++;
        }
    }
}

__kernel void repeat_F32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width, int height, int channel, int axis)
{
    int2 coord = (int2)(0, 0);
    int2 coord_out = coord;

    for(coord.x = 0; coord.x < width;)
    {
        float4 data = read_imagef(input0, coord);
        int4 len = read_imagei(input1, coord.xy);
        coord.x++;
        for(int i = 0; i < len.x; i++)
        {
            write_imagef(output, coord_out, data);
            coord_out.x++;
        }
    }
}

