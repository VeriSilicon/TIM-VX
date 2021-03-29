
__kernel void space2depth_internal_F32toF32 (
        image2d_array_t    input,
        image2d_array_t    output,
        int block_size_x, int block_size_y,
        float  scaleInOut, float zpInOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int inDepth = get_image_array_size(input);

    int4 coord = (int4)(x, y, z, 0);
    float4 data = {0.0};
    data = read_imagef(input, coord);

    ushort blockSize_x = convert_ushort(block_size_x);
    ushort blockSize_y = convert_ushort(block_size_y);
    int4 coord_out = (int4)(convert_ushort(x)/blockSize_x, convert_ushort(y)/blockSize_y, 0, 0);
    coord_out.z = ((x - coord_out.x * block_size_x) + (y - coord_out.y * block_size_y) * block_size_x) * inDepth
                     + z;
    write_imagef(output, coord_out, data);
}

__kernel void space2depth_internal_F32toF32_X2Y1 (
        image2d_array_t    input,
        image2d_array_t    output,
        int block_size_x, int block_size_y,
        float  scaleInOut, float zpInOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int inDepth = get_image_array_size(input);

    int4 coord = (int4)(x, y, z, 0);
    float4 data = {0.0};
    data = read_imagef(input, coord);

    int4 coord_out = (int4)(x >> 1, y, 0, 0);
    coord_out.z = (x & 1) * inDepth + z;
    write_imagef(output, coord_out, data);
}

__kernel void space2depth_internal_U8toU8 (
        image2d_array_t    input,
        image2d_array_t    output,
        int block_size_x, int block_size_y,
        float  scaleInOut, float zpInOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int inDepth = get_image_array_size(input);

    int4 coord = (int4)(x, y, z, 0);
    uint4 data = {0};
    data = read_imageui(input, coord);

    ushort blockSize_x = convert_ushort(block_size_x);
    ushort blockSize_y = convert_ushort(block_size_y);
    int4 coord_out = (int4)(convert_ushort(x)/blockSize_x, convert_ushort(y)/blockSize_y, 0, 0);
    coord_out.z = ((x - coord_out.x * block_size_x) + (y - coord_out.y * block_size_y) * block_size_x) * inDepth
                    + z;

    data.x = convert_uint(data.x * scaleInOut + zpInOut);
    write_imageui(output, coord_out, data);
}

__kernel void space2depth_internal_U8toU8_X2Y1 (
        image2d_array_t    input,
        image2d_array_t    output,
        int block_size_x, int block_size_y,
        float  scaleInOut, float zpInOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int inDepth = get_image_array_size(input);

    int4 coord = (int4)(x, y, z, 0);
    uint4 data = {0};
    data = read_imageui(input, coord);

    int4 coord_out = (int4)(x >> 1, y, 0, 0);
    coord_out.z = (x & 1) * inDepth + z;

    data.x = convert_uint(data.x * scaleInOut + zpInOut);
    write_imageui(output, coord_out, data);
}
