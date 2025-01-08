__kernel void rope_F32_F32toF32_axis0
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));
    float4 cos, sin;

    READ_IMAGEF_2DARRAY(cos, cos_cache, coord);
    READ_IMAGEF_2DARRAY(sin, sin_cache, coord);
    coord.x = coord.x * step;
    float4 src0 = read_imagef(input, coord);
    int4 coord_out = coord;

    coord.x += half_head_size;
    float4 src1 = read_imagef(input, coord);

    float4 dst0 = src0 * cos - src1 * sin;
    float4 dst1 = src0 * sin + src1 * cos;

    write_imagef(output, coord_out, dst0);
    coord_out.x += half_head_size;
    write_imagef(output, coord_out, dst1);
}

__kernel void rope_F32_F32toF32_axis1
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));
    float4 cos, sin;

    READ_IMAGEF_2DARRAY(cos, cos_cache, coord);
    READ_IMAGEF_2DARRAY(sin, sin_cache, coord);
    coord.y = coord.y * step;
    float4 src0 = read_imagef(input, coord);
    int4 coord_out = coord;
    coord.y += half_head_size;
    float4 src1 = read_imagef(input, coord);

    float4 dst0 = src0 * cos - src1 * sin;
    float4 dst1 = src0 * sin + src1 * cos;

    write_imagef(output, coord_out, dst0);
    coord_out.y += half_head_size;
    write_imagef(output, coord_out, dst1);
}

__kernel void rope_F32_F32toF32_axis2
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));

    float4 cos = read_imagef(cos_cache, coord);
    float4 sin = read_imagef(sin_cache, coord);
    coord.z = coord.z * step;
    float4 src0 = read_imagef(input, coord);
    int4 coord_out = coord;
    coord.z += half_head_size;
    float4 src1 = read_imagef(input, coord);

    float4 dst0 = src0 * cos - src1 * sin;
    float4 dst1 = src0 * sin + src1 * cos;

    write_imagef(output, coord_out, dst0);
    coord_out.z += half_head_size;
    write_imagef(output, coord_out, dst1);
}

__kernel void rope_I32_I32toI32_axis0
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));
    int4 _cos, _sin;
    float4 cos, sin;

    READ_IMAGEI_2DARRAY(_cos, cos_cache, coord);
    READ_IMAGEI_2DARRAY(_sin, sin_cache, coord);
    coord.x = coord.x * step;
    float4 src0 = convert_float4(read_imagei(input, coord));
    int4 coord_out = coord;

    coord.x += half_head_size;
    float4 src1 = convert_float4(read_imagei(input, coord));

    src0 = src0 - input_zp;
    src1 = src1 - input_zp;
    cos = convert_float4(_cos) - cos_zp;
    sin = convert_float4(_sin) - sin_zp;

    float4 _dst0 = src0 * cos * scale0 - src1 * sin * scale1 + output_zp;
    float4 _dst1 = src0 * sin * scale1 + src1 * cos * scale0 + output_zp;
    int4 dst0 = convert_int4_rte(_dst0);
    int4 dst1 = convert_int4_rte(_dst1);

    write_imagei(output, coord_out, dst0);
    coord_out.x += half_head_size;
    write_imagei(output, coord_out, dst1);
}

__kernel void rope_I32_I32toI32_axis1
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));
    int4 _cos, _sin;
    float4 cos, sin;

    READ_IMAGEI_2DARRAY(_cos, cos_cache, coord);
    READ_IMAGEI_2DARRAY(_sin, sin_cache, coord);
    coord.y = coord.y * step;
    float4 src0 = convert_float4(read_imagei(input, coord));
    int4 coord_out = coord;

    coord.y += half_head_size;
    float4 src1 = convert_float4(read_imagei(input, coord));

    src0 = src0 - input_zp;
    src1 = src1 - input_zp;
    cos = convert_float4(_cos) - cos_zp;
    sin = convert_float4(_sin) - sin_zp;

    float4 _dst0 = src0 * cos * scale0 - src1 * sin * scale1 + output_zp;
    float4 _dst1 = src0 * sin * scale1 + src1 * cos * scale0 + output_zp;
    int4 dst0 = convert_int4_rte(_dst0);
    int4 dst1 = convert_int4_rte(_dst1);

    write_imagei(output, coord_out, dst0);
    coord_out.y += half_head_size;
    write_imagei(output, coord_out, dst1);
}

__kernel void rope_I32_I32toI32_axis2
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));

    float4 cos = convert_float4(read_imagei(cos_cache, coord));
    float4 sin = convert_float4(read_imagei(sin_cache, coord));
    coord.z = coord.z * step;
    float4 src0 = convert_float4(read_imagei(input, coord));
    int4 coord_out = coord;

    coord.z += half_head_size;
    float4 src1 = convert_float4(read_imagei(input, coord));

    src0 = src0 - input_zp;
    src1 = src1 - input_zp;
    cos = cos - cos_zp;
    sin = sin - sin_zp;

    float4 _dst0 = src0 * cos * scale0 - src1 * sin * scale1 + output_zp;
    float4 _dst1 = src0 * sin * scale1 + src1 * cos * scale0 + output_zp;
    int4 dst0 = convert_int4_rte(_dst0);
    int4 dst1 = convert_int4_rte(_dst1);

    write_imagei(output, coord_out, dst0);
    coord_out.z += half_head_size;
    write_imagei(output, coord_out, dst1);
}

__kernel void rope_U32_U32toU32_axis0
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));
    uint4 _cos, _sin;
    float4 cos, sin;

    READ_IMAGEUI_2DARRAY(_cos, cos_cache, coord);
    READ_IMAGEUI_2DARRAY(_sin, sin_cache, coord);
    coord.x = coord.x * step;
    float4 src0 = convert_float4(read_imageui(input, coord));
    int4 coord_out = coord;

    coord.x += half_head_size;
    float4 src1 = convert_float4(read_imageui(input, coord));

    src0 = src0 - input_zp;
    src1 = src1 - input_zp;
    cos = convert_float4(_cos) - cos_zp;
    sin = convert_float4(_sin) - sin_zp;

    float4 _dst0 = src0 * cos * scale0 - src1 * sin * scale1 + output_zp;
    float4 _dst1 = src0 * sin * scale1 + src1 * cos * scale0 + output_zp;
    uint4 dst0 = convert_uint4_rte(_dst0);
    uint4 dst1 = convert_uint4_rte(_dst1);

    write_imageui(output, coord_out, dst0);
    coord_out.x += half_head_size;
    write_imageui(output, coord_out, dst1);
}

__kernel void rope_U32_U32toU32_axis1
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));
    uint4 _cos, _sin;
    float4 cos, sin;

    READ_IMAGEUI_2DARRAY(_cos, cos_cache, coord);
    READ_IMAGEUI_2DARRAY(_sin, sin_cache, coord);
    coord.y = coord.y * step;
    float4 src0 = convert_float4(read_imageui(input, coord));
    int4 coord_out = coord;

    coord.y += half_head_size;
    float4 src1 = convert_float4(read_imageui(input, coord));

    src0 = src0 - input_zp;
    src1 = src1 - input_zp;
    cos = convert_float4(_cos) - cos_zp;
    sin = convert_float4(_sin) - sin_zp;

    float4 _dst0 = src0 * cos * scale0 - src1 * sin * scale1 + output_zp;
    float4 _dst1 = src0 * sin * scale1 + src1 * cos * scale0 + output_zp;
    uint4 dst0 = convert_uint4_rte(_dst0);
    uint4 dst1 = convert_uint4_rte(_dst1);

    write_imageui(output, coord_out, dst0);
    coord_out.y += half_head_size;
    write_imageui(output, coord_out, dst1);
}

__kernel void rope_U32_U32toU32_axis2
  (
    __read_only  image2d_array_t input,
    __read_only  image2d_array_t cos_cache,
    __read_only  image2d_array_t sin_cache,
    __write_only image2d_array_t output,
                 int axis,
                 float input_zp,
                 float cos_zp,
                 float sin_zp,
                 float scale0,
                 float scale1,
                 float output_zp,
                 int half_head_size,
                 int step
  )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(2));

    float4 cos = convert_float4(read_imageui(cos_cache, coord));
    float4 sin = convert_float4(read_imageui(sin_cache, coord));
    coord.z = coord.z * step;
    float4 src0 = convert_float4(read_imageui(input, coord));
    int4 coord_out = coord;

    coord.z += half_head_size;
    float4 src1 = convert_float4(read_imageui(input, coord));

    src0 = src0 - input_zp;
    src1 = src1 - input_zp;
    cos = cos - cos_zp;
    sin = sin - sin_zp;

    float4 _dst0 = src0 * cos * scale0 - src1 * sin * scale1 + output_zp;
    float4 _dst1 = src0 * sin * scale1 + src1 * cos * scale0 + output_zp;
    uint4 dst0 = convert_uint4_rte(_dst0);
    uint4 dst1 = convert_uint4_rte(_dst1);

    write_imageui(output, coord_out, dst0);
    coord_out.z += half_head_size;
    write_imageui(output, coord_out, dst1);
}
