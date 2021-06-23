
__kernel void sequence_mask_I32toU8(
    image2d_t input, image2d_array_t output, int maxLen,
    float input_scale, float input_zpScale, float outputVal1, int output_ZP)
{
    int gidx = get_global_id(0);
    int4 coord = (int4)(gidx, get_global_id(1), get_global_id(2), 0);
    int4 index = read_imagei(input, coord.yz);
    uint4 data;
    data.x = gidx < index.x ? convert_uint_rte(outputVal1) : (uint)(output_ZP);
    write_imageui(output, coord, data);
}

__kernel void sequence_mask_I32toU8_2D(
    image2d_t input, image2d_t output, int maxLen,
    float input_scale, float input_zpScale, float outputVal1, int output_ZP)
{
    int gidx = get_global_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));
    int4 index = read_imagei(input, coord.yy);
    uint4 data;
    data.x = gidx < index.x ? convert_uint_rte(outputVal1) : (uint)(output_ZP);
    write_imageui(output, coord, data);
}

__kernel void sequence_mask_I32toI32(
    image2d_t input, image2d_array_t output, int maxLen,
    float input_scale, float input_zpScale, float outputVal1, int output_ZP)
{
    int gidx = get_global_id(0);
    int4 coord = (int4)(gidx, get_global_id(1), get_global_id(2), 0);
    int4 index = read_imagei(input, coord.yz);
    int4 data;
    data = gidx < index.x ? (int4)(1) : (int4)(0);
    write_imagei(output, coord, data);
}

__kernel void sequence_mask_I32toI32_2D(
    image2d_t input, image2d_t output, int maxLen,
    float input_scale, float input_zpScale, float outputVal1, int output_ZP)
{
    int gidx = get_global_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));
    int4 index = read_imagei(input, coord.yy);
    int4 data;
    data = gidx < index.x ? (int4)(1) : (int4)(0);
    write_imagei(output, coord, data);
}

__kernel void sequence_mask_I32toF32(
    image2d_t input, image2d_array_t output, int maxLen,
    float input_scale, float input_zpScale, float outputVal1, int output_ZP)
{
    int gidx = get_global_id(0);
    int4 coord = (int4)(gidx, get_global_id(1), get_global_id(2), 0);
    int4 index = read_imagei(input, coord.yz);
    float4 data;
    data = gidx < index.x ? (float4)(1.0f) : (float4)(0.0f);
    write_imagef(output, coord, data);
}

__kernel void sequence_mask_I32toF32_2D(
    image2d_t input, image2d_t output, int maxLen,
    float input_scale, float input_zpScale, float outputVal1, int output_ZP)
{
    int gidx = get_global_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));
    int4 index = read_imagei(input, coord.yy);
    float4 data;
    data = gidx < index.x ? (float4)(1.0f) : (float4)(0.0f);
    write_imagef(output, coord, data);
}