
#define NEAREST_INDEX_PROCESS() \
    int4   coord_out  = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    float  in_x       = (convert_float(coord_out.x) + half_pixel_value) * scale_x + round_value; \
    int    in_x_idx   = convert_int(in_x); \
    float  in_y       = (convert_float(coord_out.y) + half_pixel_value) * scale_y + round_value; \
    int    in_y_idx   = convert_int(in_y); \
    float  in_z       = (convert_float(coord_out.z) + half_pixel_value) * scale_z + round_value; \
    int    in_z_idx   = convert_int(in_z); \

__kernel void resize_3d_nearest_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           float  round_value,
                           float  output_scale,
                           float  output_tail)
{
    NEAREST_INDEX_PROCESS()
    int4 coord_in = (int4)(in_x_idx, in_y_idx, in_z_idx, 0);
    float4 dst;
    dst    = read_imagef(input, coord_in);
    write_imagef(output, coord_out, dst);
}


__kernel void resize_3d_nearest_U8toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           float  round_value,
                           float  output_scale,
                           float  output_tail)
{
    NEAREST_INDEX_PROCESS()
    int4 coord_in = (int4)(in_x_idx, in_y_idx, in_z_idx, 0);
    uint4 dst;
    dst    = convert_uint4(convert_float4(read_imageui(input, coord_in)) * output_scale + output_tail);
    write_imageui(output, coord_out, dst);
}

__kernel void resize_3d_nearest_U8toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           float  round_value,
                           float  output_scale,
                           float  output_tail)
{
    NEAREST_INDEX_PROCESS()
    int4 coord_in = (int4)(in_x_idx, in_y_idx, in_z_idx, 0);
    float4 dst;
    dst    = convert_float4(read_imageui(input, coord_in)) * output_scale + output_tail;
    write_imagef(output, coord_out, dst);
}

__kernel void resize_3d_nearest_F32toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           float  round_value,
                           float  output_scale,
                           float  output_tail)
{
    NEAREST_INDEX_PROCESS()
    int4 coord_in = (int4)(in_x_idx, in_y_idx, in_z_idx, 0);
    uint4 dst;
    dst    = convert_uint4(read_imagef(input, coord_in) * output_scale + output_tail);
    write_imageui(output, coord_out, dst);
}

__kernel void resize_3d_nearest_I8toI8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           float  round_value,
                           float  output_scale,
                           float  output_tail)
{
    NEAREST_INDEX_PROCESS()
    int4 coord_in = (int4)(in_x_idx, in_y_idx, in_z_idx, 0);
    int4 dst;
    dst    = convert_int4(convert_float4(read_imagei(input, coord_in)) * output_scale);
    write_imagei(output, coord_out, dst);
}

__kernel void resize_3d_nearest_BF16toBF16(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           float  round_value,
                           float  output_scale,
                           float  output_tail)
{
    NEAREST_INDEX_PROCESS()
    int4 coord_in = (int4)(in_x_idx, in_y_idx, in_z_idx, 0);
    uint4 dst;
    dst = read_imageui(input, coord_in);
    write_imageui(output, coord_out, dst);
}

