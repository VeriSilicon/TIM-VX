__kernel void bilinear_grid_sample_F32_F32toF32(
    __read_only  image2d_array_t  input0,
    __read_only  image2d_t        input1,
    __write_only image2d_array_t  output,
                           float  half_input0_w,
                           float  half_input0_h,
                           float  add_float_value_w,
                           float  add_float_value_h,
                           int    depth
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int2   coord_in1    =  (int2)(get_global_id(0) * 2, get_global_id(1));
    int2   coord_add    = (int2)(-1, 1);

    float fx = read_imagef(input1, coord_in1).x;
    coord_in1.x = coord_in1.x + 1;
    float fy = read_imagef(input1, coord_in1).x;

    fx = fx * half_input0_w + add_float_value_w;
    fy = fy * half_input0_h + add_float_value_h;
    float x_f = floor(fx);
    float y_f = floor(fy);
    float x_lerp  = fx - x_f;
    float y_lerp  = fy - y_f;
    int   x_index = convert_int(x_f);
    int   y_index = convert_int(y_f);
    int4   coord_in     = (int4)(x_index, y_index, 0, 0);

    float4 top_l, top_r, bottom_l, bottom_r, top, bottom, dst;

    while (coord_in.z < depth){
        top_l    = read_imagef(input0, coord_in);
        coord_in.y++;
        bottom_l = read_imagef(input0, coord_in);
        coord_in.x++;
        bottom_r = read_imagef(input0, coord_in);
        coord_in.y--;
        top_r    = read_imagef(input0, coord_in);
        top_r    = top_r - top_l;
        top      = top_l + x_lerp * top_r;
        bottom_r = bottom_r - bottom_l;
        bottom   = bottom_l + x_lerp * bottom_r;
        bottom   = bottom - top;
        dst      = top + y_lerp * bottom;
        write_imagef(output, coord_out, dst);
        coord_in.xz = coord_in.xz + coord_add;
        coord_out.z++;
    }
}


__kernel void bilinear_grid_sample_U8_U8toU8(
    __read_only  image2d_array_t  input0,
    __read_only  image2d_t        input1,
    __write_only image2d_array_t  output,
                           float  half_input0_w,
                           float  half_input0_h,
                           float  add_float_value_w,
                           float  add_float_value_h,
                           int    depth,
                           float  in0_scale,
                           float  in0_tail,
                           float  in1_scale,
                           float  in1_tail,
                           float  out_scale,
                           float  out_tail
                           )
{
    int4   coord_out    =  (int4)(get_global_id(0), get_global_id(1), 0, 0);
    int2   coord_in1    =  (int2)(get_global_id(0) * 2, get_global_id(1));
    int2   coord_add    = (int2)(-1, 1);

    float fx    = convert_float4(read_imageui(input1, coord_in1)).x * in1_scale + in1_tail;
    coord_in1.x = coord_in1.x + 1;
    float fy    = convert_float4(read_imageui(input1, coord_in1)).x * in1_scale + in1_tail;

    fx = fx * half_input0_w + add_float_value_w;
    fy = fy * half_input0_h + add_float_value_h;
    float x_f = floor(fx);
    float y_f = floor(fy);
    float x_lerp  = fx - x_f;
    float y_lerp  = fy - y_f;
    int   x_index = convert_int(x_f);
    int   y_index = convert_int(y_f);
    int4   coord_in     = (int4)(x_index, y_index, 0, 0);

    float4 top_l, top_r, bottom_l, bottom_r, top, bottom;
    uint4  dst;

    while (coord_in.z < depth){
        top_l    = convert_float4(read_imageui(input0, coord_in)) * in0_scale + in0_tail;
        coord_in.y++;
        bottom_l = convert_float4(read_imageui(input0, coord_in)) * in0_scale + in0_tail;
        coord_in.x++;
        bottom_r = convert_float4(read_imageui(input0, coord_in)) * in0_scale + in0_tail;
        coord_in.y--;
        top_r    = convert_float4(read_imageui(input0, coord_in)) * in0_scale + in0_tail;
        top_r    = top_r - top_l;
        top      = top_l + x_lerp * top_r;
        bottom_r = bottom_r - bottom_l;
        bottom   = bottom_l + x_lerp * bottom_r;
        bottom   = bottom - top;
        top      = top + y_lerp * bottom;
        dst      = convert_uint4_rte(top * out_scale + out_tail);
        write_imageui(output, coord_out, dst);
        coord_in.xz = coord_in.xz + coord_add;
        coord_out.z++;
    }

}