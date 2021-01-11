__kernel void argmax_axis2_F32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    float4 minVal = read_imagef(input, coord);
    int minIdx = 0;
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minIdx = val.x > minVal.x ? coord.z : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.z ++;
    }

    write_imagei(output, coord.xy, minIdx);
}

__kernel void argmax_axis2_F32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int minIdx = 0;

    write_imagei(output, coord.xy, minIdx);
}

__kernel void argmax_axis2_U8toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    uint4 minVal = read_imageui(input, coord);
    int minIdx = 0;
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minIdx = val.x > minVal.x ? coord.z : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.z ++;
    }

    write_imagei(output, coord.xy, minIdx);
}

__kernel void argmax_axis2_U8toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int minIdx = 0;

    write_imagei(output, coord.xy, minIdx);
}

__kernel void argmax_axis2_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                         int     axisSize
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    int4 minVal = read_imagei(input, coord);
    int minIdx = 0;
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minIdx = val.x > minVal.x ? coord.z : minIdx;
        minVal = val > minVal ? val : minVal;
        coord.z ++;
    }

    write_imagei(output, coord.xy, minIdx);
}

__kernel void argmax_axis2_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                       int axisSize
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int minIdx = 0;

    write_imagei(output, coord.xy, minIdx);
}
