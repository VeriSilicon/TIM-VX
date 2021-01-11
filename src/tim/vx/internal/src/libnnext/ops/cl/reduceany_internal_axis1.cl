__kernel void reduceany_axis1_I8toI8
    (
    __read_only  image2d_array_t input,
    __write_only image2d_t       output
    )
{
    int4 coord =  (int4)(get_global_id(0), 0, get_global_id(1), 0);
    int axisSize = get_image_height(input);

    int4 anyVal = read_imagei(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        anyVal = val || anyVal;
        coord.y ++;
    }
    anyVal.x = anyVal.x & 1;
    write_imagei(output, coord.xz, anyVal);
}

__kernel void reduceany_axis1_I8toI8_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output
    )
{
    int2 coord =  (int2)(get_global_id(0), 0);
    int axisSize = get_image_height(input);

    int4 anyVal = read_imagei(input, coord);
    coord.y ++;

    for (; coord.y < axisSize;)
    {
        int4 val = read_imagei(input, coord);
        anyVal = val || anyVal;
        coord.y ++;
    }
    anyVal.x = anyVal.x & 1;
    coord.y = 0;
    write_imagei(output, coord, anyVal);
}

