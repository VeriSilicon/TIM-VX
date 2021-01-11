__kernel void reducemin_axis1_F32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);

    float4 minVal = read_imagef(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.y ++;
    }

    write_imagef(output, coord.xz, minVal);
}

__kernel void reducemin_axis1_F32toF32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);

    float4 minVal = read_imagef(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagef(output, coord, minVal);
}

__kernel void reducemin_axis1_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);
    uint4 dst;
    uint4 minVal = read_imageui(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.y ++;
    }
    dst = convert_uint4(convert_float4(minVal) * inputScale + inputTail);
    write_imageui(output, coord.xz, dst);
}

__kernel void reducemin_axis1_U8toU8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);
    uint4 dst;
    uint4 minVal = read_imageui(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.y ++;
    }
    dst = convert_uint4(convert_float4(minVal) * inputScale + inputTail);
    coord.y = 0;
    write_imageui(output, coord, dst);
}

__kernel void reducemin_axis1_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);

    int4 minVal = read_imagei(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.y ++;
    }

    write_imagei(output, coord.xz, minVal);
}

__kernel void reducemin_axis1_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                 float     inputScale,
                 float     inputTail
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);

    int4 minVal = read_imagei(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.y ++;
    }

    coord.y = 0;
    write_imagei(output, coord, minVal);
}

