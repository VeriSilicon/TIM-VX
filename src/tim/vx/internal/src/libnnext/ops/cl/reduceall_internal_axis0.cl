__kernel void reduceall_axis0_I8toI8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord   =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    int axisSize = get_image_width(input);

    int4 allVal  = read_imagei(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        allVal = val && allVal;
        coord.x ++;
    }
    allVal.x = allVal.x & 1;
    write_imagei(output, coord.yz, allVal);
}

__kernel void reduceall_axis0_I8toI8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output
    )
{
    int2 coord   =  (int2)(0, get_global_id(0));
    int axisSize = get_image_width(input);

    int4 allVal  = read_imagei(input, coord);
    coord.x ++;

    for (; coord.x < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        allVal = val && allVal;
        coord.x ++;
    }
    allVal.x = allVal.x & 1;
    coord.x = 0;
    write_imagei(output, coord.yx, allVal);
}

