__kernel void gather_nd_batch_U8toU8_1D(
    __read_only image2d_t   input0,
    __read_only image2d_array_t   input1,
    __write_only image2d_array_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // index_num
    int gidz = get_global_id(2);  // batch_num

    int4 coord = (int4)(gidx, gidy, gidz, 0);
    int4 indice = read_imagei(input1, coord.wyzw);
    int2 coord0 = (int2)(indice.x * block_size + gidx, gidz);

    uint4 data = read_imageui(input0, coord0);
    write_imageui(output, coord, data);
}

__kernel void gather_nd_batch_F16toF16_1D(
    __read_only image2d_t   input0,
    __read_only image2d_array_t   input1,
    __write_only image2d_array_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // index_num
    int gidz = get_global_id(2);  // batch_num

    int4 coord = (int4)(gidx, gidy, gidz, 0);
    int4 indice = read_imagei(input1, coord.wyzw);
    int2 coord0 = (int2)(indice.x * block_size + gidx, gidz);

    float4 data = read_imagef(input0, coord0);
    write_imagef(output, coord, data);
}

__kernel void gather_nd_batch_I8toI8_1D(
    __read_only image2d_t   input0,
    __read_only image2d_array_t   input1,
    __write_only image2d_array_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // index_num
    int gidz = get_global_id(2);  // batch_num

    int4 coord = (int4)(gidx, gidy, gidz, 0);
    int4 indice = read_imagei(input1, coord.wyzw);
    int2 coord0 = (int2)(indice.x * block_size + gidx, gidz);

    int4 data = read_imagei(input0, coord0);
    write_imagei(output, coord, data);
}

//2D
__kernel void gather_nd_batch_U8toU8_2D(
    __read_only image2d_array_t   input0,
    __read_only image2d_array_t   input1,
    __write_only image2d_array_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // index_num
    int gidz = get_global_id(2);  // batch_num

    int4 coord = (int4)(1, gidy, gidz, 0);
    int4 indice = read_imagei(input1, coord.wyzw);
    int4 indice1 = read_imagei(input1, coord.xyzw);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;
    indice.zw = coord.zw;

    uint4 data = read_imageui(input0, indice);
    coord.x = gidx;
    write_imageui(output, coord, data);
}

__kernel void gather_nd_batch_F16toF16_2D(
    __read_only image2d_array_t   input0,
    __read_only image2d_array_t   input1,
    __write_only image2d_array_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // index_num
    int gidz = get_global_id(2);  // batch_num

    int4 coord = (int4)(1, gidy, gidz, 0);
    int4 indice = read_imagei(input1, coord.wyzw);
    int4 indice1 = read_imagei(input1, coord.xyzw);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;
    indice.zw = coord.zw;

    float4 data = read_imagef(input0, indice);
    coord.x = gidx;
    write_imagef(output, coord, data);
}

__kernel void gather_nd_batch_I8toI8_2D(
    __read_only image2d_array_t   input0,
    __read_only image2d_array_t   input1,
    __write_only image2d_array_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // index_num
    int gidz = get_global_id(2);  // batch_num

    int4 coord = (int4)(1, gidy, gidz, 0);
    int4 indice = read_imagei(input1, coord.wyzw);
    int4 indice1 = read_imagei(input1, coord.xyzw);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;
    indice.y = indice1.x;
    indice.zw = coord.zw;

    int4 data = read_imagei(input0, indice);
    coord.x = gidx;
    write_imagei(output, coord, data);
}
