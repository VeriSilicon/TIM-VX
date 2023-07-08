#pragma OPENCL EXTENSION CL_VIV_asm : enable

#define RESIZE_3D(in_name, out_name, read_image_type, dst_type, convert_type, write_image_type) \
__kernel void resize_3d_bilinear_##in_name##to##out_name( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output, \
                           float  scale_x, \
                           float  scale_y, \
                           float  scale_z, \
                           float  half_pixel_value, \
                           uint   in_width, \
                           uint   in_height, \
                           uint   in_depth, \
                           float  in_scale, \
                           float  in_tail, \
                           float  out_scale, \
                           float  out_tail \
                           ) \
{ \
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    float  in_x         = (convert_float(coord_out.x) + half_pixel_value) * scale_x - half_pixel_value; \
    float  left_x_f     = fmax(floor(in_x), 0); \
    float  x_lerp       = in_x - left_x_f; \
    int    left_x_idx   = convert_int(left_x_f); \
    float  in_y         = (convert_float(coord_out.y) + half_pixel_value) * scale_y - half_pixel_value; \
    float  top_y_f      = fmax(floor(in_y), 0); \
    float  y_lerp       = in_y - top_y_f; \
    int    top_y_idx    = convert_int(top_y_f); \
    float  in_z         = (convert_float(coord_out.z) + half_pixel_value) * scale_z - half_pixel_value; \
    float  front_z_f    = fmax(floor(in_z), 0); \
    float  z_lerp       = in_z - front_z_f; \
    int    front_z_idx  = convert_int(front_z_f); \
    int4   coord_in     = (int4)(left_x_idx, top_y_idx, front_z_idx, 0); \
    float4 data_000, data_100, data_010, data_110, data_001, data_011, data_101, data_111; \
    dst_type  dst; \
 \
    int dx, dy, dz; \
    dx = in_x < 0 ? 0 : (left_x_f < in_width - 1 ? 1 : 0); \
    dy = in_y < 0 ? 0 : (top_y_f < in_height - 1 ? 1 : 0); \
    dz = in_z < 0 ? 0 : (front_z_idx < in_depth - 1 ? 1 : 0); \
 \
    data_000 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.y = coord_in.y + dy; \
    data_010 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.x = coord_in.x + dx; \
    data_110 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.y = coord_in.y - dy; \
    data_100 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.z = coord_in.z + dz; \
    data_101 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.y = coord_in.y + dy; \
    data_111 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.x = coord_in.x - dx; \
    data_011 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
    coord_in.y = coord_in.y - dy; \
    data_001 = convert_float4(read_image_type(input, coord_in)) * in_scale + in_tail; \
 \
    data_000 = data_000 + (data_100 - data_000) * x_lerp; \
    data_010 = data_010 + (data_110 - data_010) * x_lerp; \
    data_000 = data_000 + (data_010 - data_000) * y_lerp; \
 \
    data_001 = data_001 + (data_101 - data_001) * x_lerp; \
    data_011 = data_011 + (data_111 - data_011) * x_lerp; \
    data_001 = data_001 + (data_011 - data_001) * y_lerp; \
    data_000 = data_000 + (data_001 - data_000) * z_lerp; \
 \
    dst      = convert_type(data_000 * out_scale + out_tail); \
 \
    write_image_type(output, coord_out, dst); \
}
RESIZE_3D(F32, F32, read_imagef,  float4, convert_float4, write_imagef)
RESIZE_3D(F32, U8,  read_imagef,  uint4,  convert_uint4,  write_imageui)
RESIZE_3D(U8,  F32, read_imageui, float4, convert_float4, write_imagef)
RESIZE_3D(U8,  U8,  read_imageui, uint4,  convert_uint4,  write_imageui)
RESIZE_3D(I8,  I8,  read_imagei,  int4,   convert_int4,   write_imagei)

__kernel void resize_3d_bilinear_BF16toBF16(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  scale_z,
                           float  half_pixel_value,
                           uint   in_width,
                           uint   in_height,
                           uint   in_depth,
                           float  in_scale,
                           float  in_tail,
                           float  out_scale,
                           float  out_tail
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float  in_x         = (convert_float(coord_out.x) + half_pixel_value) * scale_x - half_pixel_value;
    float  left_x_f     = fmax(floor(in_x), 0);
    float  x_lerp       = in_x - left_x_f;
    int    left_x_idx   = convert_int(left_x_f);
    float  in_y         = (convert_float(coord_out.y) + half_pixel_value) * scale_y - half_pixel_value;
    float  top_y_f      = fmax(floor(in_y), 0);
    float  y_lerp       = in_y - top_y_f;
    int    top_y_idx    = convert_int(top_y_f);
    float  in_z         = (convert_float(coord_out.z) + half_pixel_value) * scale_z - half_pixel_value;
    float  front_z_f    = fmax(floor(in_z), 0);
    float  z_lerp       = in_z - front_z_f;
    int    front_z_idx  = convert_int(front_z_f);
    int4   coord_in     = (int4)(left_x_idx, top_y_idx, front_z_idx, 0);
    uint4 data_000, data_100, data_010, data_110, data_001, data_011, data_101, data_111;
    float4 data_000_f, data_100_f, data_010_f, data_110_f, data_001_f, data_011_f, data_101_f, data_111_f;
    uint4  dst;

    int dx, dy, dz;
    dx = in_x < 0 ? 0 : (left_x_f < in_width - 1 ? 1 : 0);
    dy = in_y < 0 ? 0 : (top_y_f < in_height - 1 ? 1 : 0);
    dz = in_z < 0 ? 0 : (front_z_idx < in_depth - 1 ? 1 : 0);

    data_000 = read_imageui(input, coord_in);
    data_000 = data_000 << 16;
    coord_in.y = coord_in.y + dy;
    data_010 = read_imageui(input, coord_in);
    data_010 = data_010 << 16;
    coord_in.x = coord_in.x + dx;
    data_110 = read_imageui(input, coord_in);
    data_110 = data_110 << 16;
    coord_in.y = coord_in.y - dy;
    data_100 = read_imageui(input, coord_in);
    data_100 = data_100 << 16;
    coord_in.z = coord_in.z + dz;
    data_101 = read_imageui(input, coord_in);
    data_101 = data_101 << 16;
    coord_in.y = coord_in.y + dy;
    data_111 = read_imageui(input, coord_in);
    data_111 = data_111 << 16;
    coord_in.x = coord_in.x - dx;
    data_011 = read_imageui(input, coord_in);
    data_011 = data_011 << 16;
    coord_in.y = coord_in.y - dy;
    data_001 = read_imageui(input, coord_in);
    data_001 = data_001 << 16;

    _viv_asm(COPY, data_000_f, data_000, 16);
    _viv_asm(COPY, data_010_f, data_010, 16);
    _viv_asm(COPY, data_110_f, data_110, 16);
    _viv_asm(COPY, data_100_f, data_100, 16);
    _viv_asm(COPY, data_101_f, data_101, 16);
    _viv_asm(COPY, data_111_f, data_111, 16);
    _viv_asm(COPY, data_011_f, data_011, 16);
    _viv_asm(COPY, data_001_f, data_001, 16);

    data_000_f = data_000_f + (data_100_f - data_000_f) * x_lerp;
    data_010_f = data_010_f + (data_110_f - data_010_f) * x_lerp;
    data_000_f = data_000_f + (data_010_f - data_000_f) * y_lerp;

    data_001_f = data_001_f + (data_101_f - data_001_f) * x_lerp;
    data_011_f = data_011_f + (data_111_f - data_011_f) * x_lerp;
    data_001_f = data_001_f + (data_011_f - data_001_f) * y_lerp;
    data_000_f = data_000_f + (data_001_f - data_000_f) * z_lerp;

    _viv_asm(COPY, dst, data_000_f, 16);
    dst = dst >> 16;
    write_imageui(output, coord_out, dst);
}
