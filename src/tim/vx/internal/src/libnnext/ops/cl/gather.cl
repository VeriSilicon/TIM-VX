__kernel void gather_U8toU8(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);

    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    uint4 data = read_imageui(input0, coord_in.zw);

    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    write_imageui(output, coord, data);
}

__kernel void gather_F16toF16(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);

    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    float4 data = read_imagef(input0, coord_in.zw);

    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    write_imagef(output, coord, data);
}

__kernel void gather_I32toI32(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);

    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    int4 data = read_imagei(input0, coord_in.zw);

    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    write_imagei(output, coord, data);
}

__kernel void gather_F32toF32(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);

    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    float4 data = read_imagef(input0, coord_in.zw);

    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    write_imagef(output, coord, data);
}
