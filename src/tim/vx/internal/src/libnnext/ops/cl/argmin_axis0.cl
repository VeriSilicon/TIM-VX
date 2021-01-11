__kernel void argmin_axis0_F32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(0, get_global_id(0), get_global_id(1), 0);

    float4 minVal = read_imagef(input, coord);
    int minIdx = 0;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minIdx = val.x < minVal.x ? coord.x : minIdx;
        minVal = val < minVal ? val : minVal;
        coord.x ++;
    }

    write_imagei(output, coord.yz, minIdx);
}

__kernel void argmin_axis0_F32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(0, get_global_id(0));

    float4 minVal = read_imagef(input, coord);
    int minIdx = 0;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minIdx = val.x < minVal.x ? coord.x : minIdx;
        minVal = val < minVal ? val : minVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagei(output, coord.yx, minIdx);
}

__kernel void argmin_axis0_U8toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(0, get_global_id(0), get_global_id(1), 0);

    uint4 minVal = read_imageui(input, coord);
    int minIdx = 0;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minIdx = val.x < minVal.x ? coord.x : minIdx;
        minVal = val < minVal ? val : minVal;
        coord.x ++;
    }

    write_imagei(output, coord.yz, minIdx);
}

__kernel void argmin_axis0_U8toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(0, get_global_id(0));

    uint4 minVal = read_imageui(input, coord);
    int minIdx = 0;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minIdx = val.x < minVal.x ? coord.x : minIdx;
        minVal = val < minVal ? val : minVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagei(output, coord.yx, minIdx);
}

__kernel void argmin_axis0_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(0, get_global_id(0), get_global_id(1), 0);

    int4 minVal = read_imagei(input, coord);
    int minIdx = 0;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minIdx = val.x < minVal.x ? coord.x : minIdx;
        minVal = val < minVal ? val : minVal;
        coord.x ++;
    }

    write_imagei(output, coord.yz, minIdx);
}

__kernel void argmin_axis0_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(0, get_global_id(0));

    int4 minVal = read_imagei(input, coord);
    int minIdx = 0;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minIdx = val.x < minVal.x ? coord.x : minIdx;
        minVal = val < minVal ? val : minVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagei(output, coord.yx, minIdx);
}

