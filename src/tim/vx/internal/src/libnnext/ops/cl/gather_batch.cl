__kernel void gather_batch_U8toU8(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
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

    int2 coord_idx = (int2)(gidy, 0);
    int4 coord_in = (int4)(gidx, 0, 0, 0);
    int4 coord = (int4)(gidx, gidz * indices_num + gidy, 0, 0);
    for(; coord_idx.y < batch;)
    {
        int4 indice = read_imagei(input1, coord_idx);
        coord_idx.y++;
        coord_in.y = gidz * axis_num + indice.x;

        uint4 data = read_imageui(input0, coord_in);
        coord_in.z++;
        write_imageui(output, coord, data);
        coord.z++;
    }
}

__kernel void gather_batch_F16toF16(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
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

    int2 coord_idx = (int2)(gidy, 0);
    int4 coord_in = (int4)(gidx, 0, 0, 0);
    int4 coord = (int4)(gidx, gidz * indices_num + gidy, 0, 0);
    for(; coord_idx.y < batch;)
    {
        int4 indice = read_imagei(input1, coord_idx);
        coord_idx.y++;
        coord_in.y = gidz * axis_num + indice.x;

        float4 data = read_imagef(input0, coord_in);
        coord_in.z++;
        write_imagef(output, coord, data);
        coord.z++;
    }
}

__kernel void gather_batch_I32toI32(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
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

    int2 coord_idx = (int2)(gidy, 0);
    int4 coord_in = (int4)(gidx, 0, 0, 0);
    int4 coord = (int4)(gidx, gidz * indices_num + gidy, 0, 0);
    for(; coord_idx.y < batch;)
    {
        int4 indice = read_imagei(input1, coord_idx);
        coord_idx.y++;
        coord_in.y = gidz * axis_num + indice.x;

        int4 data = read_imagei(input0, coord_in);
        coord_in.z++;
        write_imagei(output, coord, data);
        coord.z++;
    }
}

__kernel void gather_batch_F32toF32(
    __read_only image2d_array_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_array_t  output,
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

    int2 coord_idx = (int2)(gidy, 0);
    int4 coord_in = (int4)(gidx, 0, 0, 0);
    int4 coord = (int4)(gidx, gidz * indices_num + gidy, 0, 0);
    for(; coord_idx.y < batch;)
    {
        int4 indice = read_imagei(input1, coord_idx);
        coord_idx.y++;
        coord_in.y = gidz * axis_num + indice.x;

        float4 data = read_imagef(input0, coord_in);
        coord_in.z++;
        write_imagef(output, coord, data);
        coord.z++;
    }
}
