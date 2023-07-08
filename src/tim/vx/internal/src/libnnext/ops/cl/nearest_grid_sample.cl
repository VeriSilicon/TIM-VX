__kernel void nearest_grid_sample_F32_F32toF32(
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

    float fx = read_imagef(input1, coord_in1).x;
    coord_in1.x = coord_in1.x + 1;
    float fy = read_imagef(input1, coord_in1).x;

    fx = fx * half_input0_w + add_float_value_w;
    fy = fy * half_input0_h + add_float_value_h;
    int   x_index = convert_int(fx);
    int   y_index = convert_int(fy);
    int4   coord_in     = (int4)(x_index, y_index, 0, 0);

    float4  dst;

    while (coord_in.z < depth){
        dst    = read_imagef(input0, coord_in);
        write_imagef(output, coord_out, dst);
        coord_in.z++;
        coord_out.z++;
    }
}


__kernel void nearest_grid_sample_U8_U8toU8(
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

    float fx    = convert_float4(read_imageui(input1, coord_in1)).x * in1_scale + in1_tail;
    coord_in1.x = coord_in1.x + 1;
    float fy    = convert_float4(read_imageui(input1, coord_in1)).x * in1_scale + in1_tail;

    fx = fx * half_input0_w + add_float_value_w;
    fy = fy * half_input0_h + add_float_value_h;
    int   x_index = convert_int(fx);
    int   y_index = convert_int(fy);
    int4   coord_in     = (int4)(x_index, y_index, 0, 0);

    float4 val;
    uint4  dst;

    while (coord_in.z < depth){
        val    = convert_float4(read_imageui(input0, coord_in)) * in0_scale + in0_tail;
        dst      = convert_uint4_rte(val * out_scale + out_tail);
        write_imageui(output, coord_out, dst);
        coord_in.z++;
        coord_out.z++;
    }

}
