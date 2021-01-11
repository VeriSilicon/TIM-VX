float exp_(float x, float logE)
{
    x *= logE;
    x = exp2(x);
    return x;
}

__kernel void detect_post_box_F32_F32toF32(
     __read_only image2d_array_t   input0,
           __read_only image2d_t   input1,
    __write_only image2d_array_t   output,
                           float   inv_scale_y,
                           float   inv_scale_x,
                           float   inv_scale_h,
                           float   inv_scale_w,
                           float   logE)
{
    int4 coord =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    float4 src0;
    float4 src1;
    float4 dst;
    float4 tmp0, tmp1;
    src0.x = read_imagef(input0, coord).x;
    src1.x = read_imagef(input1, coord.xy).x;
    coord.x++;
    src0.y = read_imagef(input0, coord).x;
    src1.y = read_imagef(input1, coord.xy).x;
    coord.x++;
    src0.z = read_imagef(input0, coord).x;
    src1.z = read_imagef(input1, coord.xy).x;
    coord.x++;
    src0.w = read_imagef(input0, coord).x;
    src1.w = read_imagef(input1, coord.xy).x;

    tmp0.x  = src1.x + src1.z * src0.x * inv_scale_y;
    tmp0.y  = src1.y + src1.w * src0.y * inv_scale_x;
    tmp1.x = src1.z * exp_(src0.z * inv_scale_h, logE) * 0.5f;
    tmp1.y = src1.w * exp_(src0.w * inv_scale_w, logE) * 0.5f;
    dst.xy = tmp0.xy - tmp1.xy;
    dst.zw = tmp0.xy + tmp1.xy;
    coord.x = 0;
    write_imagef(output, coord, dst.xxxx);
    coord.x++;
    write_imagef(output, coord, dst.yyyy);
    coord.x++;
    write_imagef(output, coord, dst.zzzz);
    coord.x++;
    write_imagef(output, coord, dst.wwww);
}


__kernel void detect_post_box_U8_U8toF32(
     __read_only image2d_array_t   input0,
           __read_only image2d_t   input1,
    __write_only image2d_array_t   output,
                           float   inv_scale_y,
                           float   inv_scale_x,
                           float   inv_scale_h,
                           float   inv_scale_w,
                           float   logE,
                           float   input0Tail,
                           float   input1Tail,
                           float   input0Scale,
                           float   input1Scale)
{
    int4 coord =  (int4)(0, get_global_id(0), get_global_id(1), 0);
    uint4  in0, in1;
    float4 src0;
    float4 src1;
    float4 dst;
    float4 tmp0, tmp1;
    in0.x = read_imageui(input0, coord).x;
    in1.x = read_imageui(input1, coord.xy).x;
    coord.x++;
    in0.y = read_imageui(input0, coord).x;
    in1.y = read_imageui(input1, coord.xy).x;
    coord.x++;
    in0.z = read_imageui(input0, coord).x;
    in1.z = read_imageui(input1, coord.xy).x;
    coord.x++;
    in0.w = read_imageui(input0, coord).x;
    in1.w = read_imageui(input1, coord.xy).x;

    src0 = convert_float4(in0) * input0Scale + input0Tail;
    src1 = convert_float4(in1) * input1Scale + input1Tail;

    tmp0.x  = src1.x + src1.z * src0.x * inv_scale_y;
    tmp0.y  = src1.y + src1.w * src0.y * inv_scale_x;
    tmp1.x = src1.z * exp_(src0.z * inv_scale_h, logE) * 0.5f;
    tmp1.y = src1.w * exp_(src0.w * inv_scale_w, logE) * 0.5f;
    dst.xy = tmp0.xy - tmp1.xy;
    dst.zw = tmp0.xy + tmp1.xy;
    coord.x = 0;
    write_imagef(output, coord, dst.xxxx);
    coord.x++;
    write_imagef(output, coord, dst.yyyy);
    coord.x++;
    write_imagef(output, coord, dst.zzzz);
    coord.x++;
    write_imagef(output, coord, dst.wwww);
}