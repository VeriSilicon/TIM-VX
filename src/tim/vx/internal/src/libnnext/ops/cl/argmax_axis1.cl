__kernel void argmax_axis1_F32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);

    float4 minVal = read_imagef(input, coord);
    int minIdx = 0;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minIdx = val.x > minVal.x ? coord.y : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.y ++;
    }

    write_imagei(output, coord.xz, minIdx);
}

__kernel void argmax_axis1_F32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);

    float4 minVal = read_imagef(input, coord);
    int minIdx = 0;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minIdx = val.x > minVal.x ? coord.y : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagei(output, coord, minIdx);
}

__kernel void argmax_axis1_U8toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);

    uint4 minVal = read_imageui(input, coord);
    int minIdx = 0;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minIdx = val.x > minVal.x ? coord.y : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.y ++;
    }

    write_imagei(output, coord.xz, minIdx);
}

__kernel void argmax_axis1_U8toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);

    uint4 minVal = read_imageui(input, coord);
    int minIdx = 0;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minIdx = val.x > minVal.x ? coord.y : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagei(output, coord, minIdx);
}

__kernel void argmax_axis1_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);

    int4 minVal = read_imagei(input, coord);
    int minIdx = 0;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minIdx = val.x > minVal.x ? coord.y : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.y ++;
    }

    write_imagei(output, coord.xz, minIdx);
}

__kernel void argmax_axis1_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);

    int4 minVal = read_imagei(input, coord);
    int minIdx = 0;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minIdx = val.x > minVal.x ? coord.y : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagei(output, coord, minIdx);
}
