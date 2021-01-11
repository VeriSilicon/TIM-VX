__kernel void reducemax_axis0_F32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord    =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize  = get_image_width(input);
    float4 maxVal = read_imagef(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        maxVal = val > maxVal ? val : maxVal;
        coord.x ++;
    }

    write_imagef(output, coord.yz, maxVal);
}

__kernel void reducemax_axis0_F32toF32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);
    float4 maxVal = read_imagef(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        maxVal = val > maxVal ? val : maxVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagef(output, coord.yx, maxVal);
}

__kernel void reducemax_axis0_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord   =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize = get_image_width(input);
    uint4 dst;
    uint4 maxVal = read_imageui(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        maxVal = val > maxVal ? val : maxVal;
        coord.x ++;
    }
    dst = convert_uint4(convert_float4(maxVal) * inputScale + inputTail);
    write_imageui(output, coord.yz, dst);
}

__kernel void reducemax_axis0_U8toU8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);
    uint4 dst;
    uint4 maxVal = read_imageui(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        maxVal = val > maxVal ? val : maxVal;
        coord.x ++;
    }
    dst = convert_uint4(convert_float4(maxVal) * inputScale + inputTail);
    coord.x = 0;
    write_imageui(output, coord.yx, dst);
}

__kernel void reducemax_axis0_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord   =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize = get_image_width(input);

    int4 maxVal  = read_imagei(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        maxVal = val > maxVal ? val : maxVal;
        coord.x ++;
    }

    write_imagei(output, coord.yz, maxVal);
}

__kernel void reducemax_axis0_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);

    int4 maxVal  = read_imagei(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        maxVal = val > maxVal ? val : maxVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagei(output, coord.yx, maxVal);
}

