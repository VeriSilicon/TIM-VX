__kernel void resize_cubic_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  half_pixel_value
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float  cubic_coeffs_y[4] = {0,0,0,0};
    float  cubic_coeffs_x[4] = {0,0,0,0};
    float  in_x         = (convert_float(coord_out.x) + half_pixel_value) * scale_x - half_pixel_value;
    float  left_x_f     = floor(in_x);
    float4 delta_x      = (float4)(0, in_x - left_x_f,0,0);
    float  in_y         = (convert_float(coord_out.y) + half_pixel_value) * scale_y - half_pixel_value;
    float  top_y_f      = floor(in_y);
    float4 delta_y      = (float4)(0, in_y - top_y_f,0,0);
    int    x_idx        = convert_int(left_x_f - 1);
    int    y_idx        = convert_int(top_y_f - 1);
    int4   coord_in     = (int4)(x_idx, y_idx, coord_out.z, 0);
    float  data00, data01, data02, data03, data10, data11, data12, data13,
           data20, data21, data22, data23, data30, data31, data32, data33;

    delta_x.x = 1 + delta_x.y;
    delta_x.z = 1 - delta_x.y;
    delta_x.w = 2 - delta_x.y;
    cubic_coeffs_x[0] = -0.5 * ((((delta_x.x - 5) * delta_x.x + 8) * delta_x.x) - 4);
    cubic_coeffs_x[1] = (1.5 * delta_x.y - 2.5) * delta_x.y * delta_x.y + 1;
    cubic_coeffs_x[2] = (1.5 * delta_x.z - 2.5) * delta_x.z * delta_x.z + 1;
    cubic_coeffs_x[3] = -0.5 * ((((delta_x.w - 5) * delta_x.w + 8) * delta_x.w) - 4);
    delta_y.x = 1 + delta_y.y;
    delta_y.z = 1 - delta_y.y;
    delta_y.w = 2 - delta_y.y;
    cubic_coeffs_y[0] = -0.5 * ((((delta_y.x - 5) * delta_y.x + 8) * delta_y.x) - 4);
    cubic_coeffs_y[1] = (1.5 * delta_y.y - 2.5) * delta_y.y * delta_y.y + 1;
    cubic_coeffs_y[2] = (1.5 * delta_y.z - 2.5) * delta_y.z * delta_y.z + 1;
    cubic_coeffs_y[3] = -0.5 * ((((delta_y.w - 5) * delta_y.w + 8) * delta_y.w) - 4);
    float4 dst = (float4)(0,0,0,0);

    data00   = read_imagef(input, coord_in).x;
    coord_in.x++;
    data10   = read_imagef(input, coord_in).x;
    coord_in.x++;
    data20   = read_imagef(input, coord_in).x;
    coord_in.x++;
    data30   = read_imagef(input, coord_in).x;

    coord_in.y++;
    data31   = read_imagef(input, coord_in).x;
    coord_in.x--;
    data21   = read_imagef(input, coord_in).x;
    coord_in.x--;
    data11   = read_imagef(input, coord_in).x;
    coord_in.x--;
    data01   = read_imagef(input, coord_in).x;

    coord_in.y++;
    data02   = read_imagef(input, coord_in).x;
    coord_in.x++;
    data12   = read_imagef(input, coord_in).x;
    coord_in.x++;
    data22   = read_imagef(input, coord_in).x;
    coord_in.x++;
    data32   = read_imagef(input, coord_in).x;

    coord_in.y++;
    data33   = read_imagef(input, coord_in).x;
    coord_in.x--;
    data23   = read_imagef(input, coord_in).x;
    coord_in.x--;
    data13   = read_imagef(input, coord_in).x;
    coord_in.x--;
    data03   = read_imagef(input, coord_in).x;

    dst.x = data00 * cubic_coeffs_x[0] * cubic_coeffs_y[0]
          + data01 * cubic_coeffs_x[0] * cubic_coeffs_y[1]
          + data02 * cubic_coeffs_x[0] * cubic_coeffs_y[2]
          + data03 * cubic_coeffs_x[0] * cubic_coeffs_y[3]
          + data10 * cubic_coeffs_x[1] * cubic_coeffs_y[0]
          + data11 * cubic_coeffs_x[1] * cubic_coeffs_y[1]
          + data12 * cubic_coeffs_x[1] * cubic_coeffs_y[2]
          + data13 * cubic_coeffs_x[1] * cubic_coeffs_y[3]
          + data20 * cubic_coeffs_x[2] * cubic_coeffs_y[0]
          + data21 * cubic_coeffs_x[2] * cubic_coeffs_y[1]
          + data22 * cubic_coeffs_x[2] * cubic_coeffs_y[2]
          + data23 * cubic_coeffs_x[2] * cubic_coeffs_y[3]
          + data30 * cubic_coeffs_x[3] * cubic_coeffs_y[0]
          + data31 * cubic_coeffs_x[3] * cubic_coeffs_y[1]
          + data32 * cubic_coeffs_x[3] * cubic_coeffs_y[2]
          + data33 * cubic_coeffs_x[3] * cubic_coeffs_y[3];

    write_imagef(output, coord_out, dst);

}


__kernel void resize_cubic_U8toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  scale_x,
                           float  scale_y,
                           float  half_pixel_value,
                           float  in_scale,
                           float  in_tail,
                           float  out_scale,
                           float  out_tail
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float  cubic_coeffs_y[4] = {0,0,0,0};
    float  cubic_coeffs_x[4] = {0,0,0,0};
    float  in_x         = (convert_float(coord_out.x) + half_pixel_value) * scale_x - half_pixel_value;
    float  left_x_f     = floor(in_x);
    float4 delta_x      = (float4)(0, in_x - left_x_f,0,0);
    float  in_y         = (convert_float(coord_out.y) + half_pixel_value) * scale_y - half_pixel_value;
    float  top_y_f      = floor(in_y);
    float4 delta_y      = (float4)(0, in_y - top_y_f,0,0);
    int    x_idx        = convert_int(left_x_f - 1);
    int    y_idx        = convert_int(top_y_f - 1);
    int4   coord_in     = (int4)(x_idx, y_idx, coord_out.z, 0);
    float  data00, data01, data02, data03, data10, data11, data12, data13,
           data20, data21, data22, data23, data30, data31, data32, data33;

    delta_x.x = 1 + delta_x.y;
    delta_x.z = 1 - delta_x.y;
    delta_x.w = 2 - delta_x.y;
    cubic_coeffs_x[0] = -0.5 * ((((delta_x.x - 5) * delta_x.x + 8) * delta_x.x) - 4);
    cubic_coeffs_x[1] = (1.5 * delta_x.y - 2.5) * delta_x.y * delta_x.y + 1;
    cubic_coeffs_x[2] = (1.5 * delta_x.z - 2.5) * delta_x.z * delta_x.z + 1;
    cubic_coeffs_x[3] = -0.5 * ((((delta_x.w - 5) * delta_x.w + 8) * delta_x.w) - 4);
    delta_y.x = 1 + delta_y.y;
    delta_y.z = 1 - delta_y.y;
    delta_y.w = 2 - delta_y.y;
    cubic_coeffs_y[0] = -0.5 * ((((delta_y.x - 5) * delta_y.x + 8) * delta_y.x) - 4);
    cubic_coeffs_y[1] = (1.5 * delta_y.y - 2.5) * delta_y.y * delta_y.y + 1;
    cubic_coeffs_y[2] = (1.5 * delta_y.z - 2.5) * delta_y.z * delta_y.z + 1;
    cubic_coeffs_y[3] = -0.5 * ((((delta_y.w - 5) * delta_y.w + 8) * delta_y.w) - 4);
    float dst = 0;
    uint4 out = (uint4)(0,0,0,0);

    data00   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x++;
    data10   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x++;
    data20   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x++;
    data30   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;

    coord_in.y++;
    data31   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x--;
    data21   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x--;
    data11   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x--;
    data01   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;

    coord_in.y++;
    data02   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x++;
    data12   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x++;
    data22   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x++;
    data32   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;

    coord_in.y++;
    data33   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x--;
    data23   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x--;
    data13   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;
    coord_in.x--;
    data03   = convert_float(read_imageui(input, coord_in).x) * in_scale + in_tail;

    dst = data00 * cubic_coeffs_x[0] * cubic_coeffs_y[0]
        + data01 * cubic_coeffs_x[0] * cubic_coeffs_y[1]
        + data02 * cubic_coeffs_x[0] * cubic_coeffs_y[2]
        + data03 * cubic_coeffs_x[0] * cubic_coeffs_y[3]
        + data10 * cubic_coeffs_x[1] * cubic_coeffs_y[0]
        + data11 * cubic_coeffs_x[1] * cubic_coeffs_y[1]
        + data12 * cubic_coeffs_x[1] * cubic_coeffs_y[2]
        + data13 * cubic_coeffs_x[1] * cubic_coeffs_y[3]
        + data20 * cubic_coeffs_x[2] * cubic_coeffs_y[0]
        + data21 * cubic_coeffs_x[2] * cubic_coeffs_y[1]
        + data22 * cubic_coeffs_x[2] * cubic_coeffs_y[2]
        + data23 * cubic_coeffs_x[2] * cubic_coeffs_y[3]
        + data30 * cubic_coeffs_x[3] * cubic_coeffs_y[0]
        + data31 * cubic_coeffs_x[3] * cubic_coeffs_y[1]
        + data32 * cubic_coeffs_x[3] * cubic_coeffs_y[2]
        + data33 * cubic_coeffs_x[3] * cubic_coeffs_y[3];
    out.x = convert_uint(dst * out_scale + out_tail);

    write_imageui(output, coord_out, out);
}
