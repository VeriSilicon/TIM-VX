__kernel void reduceprod_axis0_F32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord    =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize  = get_image_width(input);
    float4 prodVal = read_imagef(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        prodVal = val * prodVal;
        coord.x ++;
    }

    write_imagef(output, coord.yz, prodVal);
}

__kernel void reduceprod_axis0_F32toF32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);
    float4 prodVal = read_imagef(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        prodVal = val * prodVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagef(output, coord.yx, prodVal);
}

__kernel void reduceprod_axis0_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputTail
    )
{
    int4 coord   =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize = get_image_width(input);
    uint4 dst;
    float4 prodVal = convert_float4(read_imageui(input, coord));
    prodVal = prodVal * inputScale + inputTail;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = convert_float4(read_imageui(input, coord));
        val = val * inputScale + inputTail;
        prodVal = val * prodVal;
        coord.x ++;
    }
    dst = convert_uint4(prodVal * outputScale + outputTail);
    write_imageui(output, coord.yz, dst);
}

__kernel void reduceprod_axis0_U8toU8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail,
                 float     outputScale,
                 float     outputTail
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);
    uint4 dst;
    float4 prodVal = convert_float4(read_imageui(input, coord));
    prodVal = prodVal * inputScale + inputTail;
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        float4 val = convert_float4(read_imageui(input, coord));
        val = val * inputScale + inputTail;
        prodVal = val * prodVal;
        coord.x ++;
    }
    dst = convert_uint4(prodVal * outputScale + outputTail);
    coord.x = 0;
    write_imageui(output, coord.yx, dst);
}

__kernel void reduceprod_axis0_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord   =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize = get_image_width(input);

    int4 prodVal  = read_imagei(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        prodVal = val * prodVal;
        coord.x ++;
    }

    write_imagei(output, coord.yz, prodVal);
}

__kernel void reduceprod_axis0_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);

    int4 prodVal  = read_imagei(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        prodVal = val * prodVal;
        coord.x ++;
    }

    coord.x = 0;
    write_imagei(output, coord.yx, prodVal);
}

