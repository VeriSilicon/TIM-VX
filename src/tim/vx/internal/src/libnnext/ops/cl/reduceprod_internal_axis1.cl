__kernel void reduceprod_axis1_F32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);

    float4 prodVal = read_imagef(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        prodVal = val * prodVal;
        coord.y ++;
    }

    write_imagef(output, coord.xz, prodVal);
}

__kernel void reduceprod_axis1_F32toF32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);

    float4 prodVal = read_imagef(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        prodVal = val * prodVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagef(output, coord, prodVal);
}

__kernel void reduceprod_axis1_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);
    uint4 dst;
    float4 prodVal = convert_float4(read_imageui(input, coord));
    prodVal = prodVal * inputScale + inputTail;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = convert_float4(read_imageui(input, coord));
        val = val * inputScale + inputTail;
        prodVal = val * prodVal;
        coord.y ++;
    }
    dst = convert_uint4(prodVal * outputScale + outputTail);
    write_imageui(output, coord.xz, dst);
}

__kernel void reduceprod_axis1_U8toU8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail,
                 float     outputScale,
                 float     outputTail
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);
    uint4 dst;
    float4 prodVal = convert_float4(read_imageui(input, coord));
    prodVal = prodVal * inputScale + inputTail;
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = convert_float4(read_imageui(input, coord));
        val = val * inputScale + inputTail;
        prodVal = val * prodVal;
        coord.y ++;
    }
    dst = convert_uint4(prodVal * outputScale + outputTail);
    coord.y = 0;
    write_imageui(output, coord, dst);
}

__kernel void reduceprod_axis1_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);

    int4 prodVal = read_imagei(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        prodVal = val * prodVal;
        coord.y ++;
    }

    write_imagei(output, coord.xz, prodVal);
}

__kernel void reduceprod_axis1_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);

    int4 prodVal = read_imagei(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        prodVal = val * prodVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagei(output, coord, prodVal);
}

