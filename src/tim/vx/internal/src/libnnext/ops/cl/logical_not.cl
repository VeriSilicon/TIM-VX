__kernel void logical_not_I8toI8(
    __read_only image2d_array_t   input,
    __write_only image2d_array_t  output)
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 src   = read_imagei(input, coord);
    int4 dst   = !src;
    dst.x = dst.x & 1;
    write_imagei(output, coord, dst);
}

__kernel void logical_not_I8toI8_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int4 src   = read_imagei(input, coord);
    int4 dst   = !src;
    dst.x = dst.x & 1;
    write_imagei(output, coord, dst);
}
