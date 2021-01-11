__kernel void reducemin_axis2_F32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);

    float4 minVal = read_imagef(input, coord);
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.z ++;
    }

    write_imagef(output, coord.xy, minVal);
}


__kernel void reducemin_axis2_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);
    uint4 dst;
    uint4 minVal = read_imageui(input, coord);
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        uint4 val = read_imageui(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.z ++;
    }
    dst = convert_uint4(convert_float4(minVal) * inputScale + inputTail);
    write_imageui(output, coord.xy, dst);
}


__kernel void reducemin_axis2_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);

    int4 minVal = read_imagei(input, coord);
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        minVal = val < minVal ? val : minVal;
        coord.z ++;
    }

    write_imagei(output, coord.xy, minVal);
}



