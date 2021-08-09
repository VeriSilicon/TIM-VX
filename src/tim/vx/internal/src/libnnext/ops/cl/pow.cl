__kernel void pow_FP32FP32toFP32
    (
    __read_only  image2d_array_t    input0,
    __read_only  image2d_array_t    input1,
    __write_only image2d_array_t    output
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    float4 src0, src1;
    float4 dst;
    READ_IMAGEF_2DARRAY(src0, input0, coord);
    READ_IMAGEF_2DARRAY(src1, input1, coord);

    float4  s0 = sign(src0);
    int4 t0 = convert_int4(src1) & 1;
    s0 = s0 == -1 ? convert_float4(t0) == 1.0f ? -1.0f : 1.0f : s0;
    dst.x = (src0.x == 0 && src1.x == 0) ? 1.0f : (src0.x != 0 ? (s0.x * exp2(src1.x * log2(fabs(src0.x)))) : 0.0f);

    write_imagef(output, coord, dst);
}

__kernel void pow_FP32FP32toFP32_2D
    (
    __read_only  image2d_t    input0,
    __read_only  image2d_t    input1,
    __write_only image2d_t    output
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    float4 src0 = read_imagef(input0, coord);
    float4 src1 = read_imagef(input1, coord);

    float4 dst = (float4)(0);

    float4  s0 = sign(src0);
    int4 t0 = convert_int4(src1) & 1;
    s0 = s0 == -1 ? convert_float4(t0) == 1.0f ? -1.0f : 1.0f : s0;

    dst.x = (src0.x == 0 && src1.x == 0) ? 1.0f : (src0.x != 0 ? (s0.x * exp2(src1.x * log2(fabs(src0.x)))) : 0.0f);

    write_imagef(output, coord, dst);
}
