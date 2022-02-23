
__kernel void depth2space_crd_F32toF32(
    image2d_array_t input, image2d_array_t output, int block_size)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int gidz = get_global_id(2);
    int4 coord_out = (int4)(gidx, gidy, gidz, 0);
    int block_e2 = block_size * block_size;
    ushort blk = (ushort)block_size;
    int inx = (int)((ushort)gidx / blk);
    int iny = (int)((ushort)gidy / blk);
    int inz = (gidx  % block_size) + (gidy % block_size) * block_size + gidz * block_e2;
    int4 coord_in = (int4)(inx, iny, inz, 0);
    float4 data = read_imagef(input, coord_in);
    write_imagef(output, coord_out, data);
}
