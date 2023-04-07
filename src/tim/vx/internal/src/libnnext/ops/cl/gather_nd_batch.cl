__kernel void gather_nd_batch_U8toU8_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // batch_num

    int4 coord = (int4)(gidx, gidy, 0, 0);
    int4 indice = read_imagei(input1, coord.wy);
    coord.z = indice.x * block_size + gidx;

    uint4 data = read_imageui(input0, coord.zy);
    write_imageui(output, coord.xy, data);
}

__kernel void gather_nd_batch_F16toF16_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // batch_num

    int4 coord = (int4)(gidx, gidy, 0, 0);
    int4 indice = read_imagei(input1, coord.wy);
    coord.z = indice.x * block_size + gidx;

    float4 data = read_imagef(input0, coord.zy);
    write_imagef(output, coord.xy, data);
}

__kernel void gather_nd_batch_I8toI8_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // batch_num

    int4 coord = (int4)(gidx, gidy, 0, 0);
    int4 indice = read_imagei(input1, coord.wy);
    coord.z = indice.x * block_size + gidx;

    int4 data = read_imagei(input0, coord.zy);
    write_imagei(output, coord.xy, data);
}

//2D
__kernel void gather_nd_batch_U8toU8_2D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // batch_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;
    indice.zw = coord.yx;

    uint4 data = read_imageui(input0, indice);
    write_imageui(output, coord.zy, data);
}

__kernel void gather_nd_batch_F16toF16_2D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // batch_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;
    indice.zw = coord.yx;

    float4 data = read_imagef(input0, indice);
    write_imagef(output, coord.zy, data);
}

__kernel void gather_nd_batch_I8toI8_2D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // batch_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;
    indice.y = indice1.x;
    indice.zw = coord.yx;

    int4 data = read_imagei(input0, indice);
    write_imagei(output, coord.zy, data);
}
