__kernel void gather_nd_U8toU8_3D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, 1, 2);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.zy);
    int4 indice2 = read_imagei(input1, coord.wy);
    indice = (int4)(indice.x * block_size + gidx, indice1.x, indice2.x, 0);
    coord.z = gidx;

    uint4 data = read_imageui(input0, indice);
    write_imageui(output, coord.zy, data);
}

__kernel void gather_nd_F16toF16_3D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord = (int4)(0, gidy, 1, 2);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.zy);
    int4 indice2 = read_imagei(input1, coord.wy);
    indice = (int4)(indice.x * block_size + gidx, indice1.x, indice2.x, 0);
    coord.z = gidx;

    float4 data = read_imagef(input0, indice);
    write_imagef(output, coord.zy, data);
}

__kernel void gather_nd_I32toI32_3D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord = (int4)(0, gidy, 1, 2);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.zy);
    int4 indice2 = read_imagei(input1, coord.wy);
    indice = (int4)(indice.x * block_size + gidx, indice1.x, indice2.x, 0);
    coord.z = gidx;

    int4 data = read_imagei(input0, indice);
    write_imagei(output, coord.zy, data);
}

__kernel void gather_nd_F32toF32_3D(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord = (int4)(0, gidy, 1, 2);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.zy);
    int4 indice2 = read_imagei(input1, coord.wy);
    indice = (int4)(indice.x * block_size + gidx, indice1.x, indice2.x, 0);
    coord.z = gidx;

    float4 data = read_imagef(input0, indice);
    write_imagef(output, coord.zy, data);
}
