__kernel void reduceany_axis2_I8toI8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int axisSize = get_image_depth(input);

    int4 anyVal = read_imagei(input, coord);
    coord.z ++;

    for (; coord.z < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        anyVal = val || anyVal;
        coord.z ++;
    }
    anyVal.x = anyVal.x & 1;
    write_imagei(output, coord.xy, anyVal);
}



