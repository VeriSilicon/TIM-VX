__kernel void gather_nd_U8toU8_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 0);
    int4 indice = read_imagei(input1, coord.xy);
    coord.w = indice.x;

    uint4 data = read_imageui(input0, coord.zw);
    write_imageui(output, coord.zy, data);
}

__kernel void gather_nd_F16toF16_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 0);
    int4 indice = read_imagei(input1, coord.xy);
    coord.w = indice.x;

    float4 data = read_imagef(input0, coord.zw);
    write_imagef(output, coord.zy, data);
}

__kernel void gather_nd_I32toI32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 0);
    int4 indice = read_imagei(input1, coord.xy);
    coord.w = indice.x;

    int4 data = read_imagei(input0, coord.zw);
    write_imagei(output, coord.zy, data);
}

__kernel void gather_nd_F32toF32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 0);
    int4 indice = read_imagei(input1, coord.xy);
    coord.w = indice.x;

    float4 data = read_imagef(input0, coord.zw);
    write_imagef(output, coord.zy, data);
}

//2D
__kernel void gather_nd_U8toU8_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;

    uint4 data = read_imageui(input0, indice.xy);
    write_imageui(output, coord.zy, data);
}

__kernel void gather_nd_F16toF16_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;

    float4 data = read_imagef(input0, indice.xy);
    write_imagef(output, coord.zy, data);
}

__kernel void gather_nd_I32toI32_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;

    int4 data = read_imagei(input0, indice.xy);
    write_imagei(output, coord.zy, data);
}

__kernel void gather_nd_F32toF32_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int coord_dim
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 coord = (int4)(0, gidy, gidx, 1);
    int4 indice = read_imagei(input1, coord.xy);
    int4 indice1 = read_imagei(input1, coord.wy);
    indice.x = indice.x * block_size + gidx;
    indice.y = indice1.x;

    float4 data = read_imagef(input0, indice.xy);
    write_imagef(output, coord.zy, data);
}
