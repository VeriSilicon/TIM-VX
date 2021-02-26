__kernel void resize_1d_bilinear_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  half_pixel_value
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float  in_x         = (convert_float(coord_out.x) + half_pixel_value) * scale_x - half_pixel_value;
    float  left_x_f     = floor(in_x);
    float  x_lerp       = in_x - left_x_f;
    int    left_x_idx   = convert_int(left_x_f);
    int4   coord_in     = (int4)(left_x_idx, coord_out.y, coord_out.z, 0);
    float4 top_l, top_r, top, bottom, dst;

    top_l    = read_imagef(input, coord_in);
    coord_in.x++;
    top_r    = read_imagef(input, coord_in);

    top_r    = top_r - top_l;
    dst      = top_l + x_lerp * top_r;

    write_imagef(output, coord_out, dst);

}


__kernel void resize_1d_bilinear_U8toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  half_pixel_value,
                           float  in_scale,
                           float  in_tail,
                           float  out_scale,
                           float  out_tail
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float  in_x         = (convert_float(coord_out.x) + half_pixel_value) * scale_x - half_pixel_value;
    float  left_x_f     = floor(in_x);
    float  x_lerp       = in_x - left_x_f;
    int    left_x_idx   = convert_int(left_x_f);
    int4   coord_in     = (int4)(left_x_idx, coord_out.y, coord_out.z, 0);
    float4 top_l, top_r, top;
    uint4  dst;

    top_l    = convert_float4(read_imageui(input, coord_in)) * in_scale + in_tail;
    coord_in.x++;
    top_r    = convert_float4(read_imageui(input, coord_in)) * in_scale + in_tail;

    top_r    = top_r - top_l;
    top      = top_l + x_lerp * top_r;
    dst      = convert_uint4(top * out_scale + out_tail);

    write_imageui(output, coord_out, dst);
}
