__kernel void reduceprod_axis2_F32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);

    float4 prodVal = read_imagef(input, coord);
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        float4 val = read_imagef(input, coord);
        prodVal = val * prodVal;
        coord.z ++;
    }

    write_imagef(output, coord.xy, prodVal);
}


__kernel void reduceprod_axis2_U8toU8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);
    uint4 dst;
    float4 prodVal = convert_float4(read_imageui(input, coord));
    prodVal = prodVal * inputScale + inputTail;
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        float4 val = convert_float4(read_imageui(input, coord));
        val = val * inputScale + inputTail;
        prodVal = val * prodVal;
        coord.z ++;
    }
    dst = convert_uint4(prodVal * outputScale + outputTail);
    write_imageui(output, coord.xy, dst);
}


__kernel void reduceprod_axis2_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);

    int4 prodVal = read_imagei(input, coord);
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        prodVal = val * prodVal;
        coord.z ++;
    }

    write_imagei(output, coord.xy, prodVal);
}



