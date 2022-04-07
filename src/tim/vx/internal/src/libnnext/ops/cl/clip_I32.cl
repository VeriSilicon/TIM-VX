__kernel void clip_I32toI32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_array_t output,
                           float minData,
                           float maxData,
                           float inputScale,
                           float inputTail,
                           float outputScale,
                           float outputZP
     )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = convert_float4(read_imagei(input, coord)) * inputScale + inputTail;
    float4 result = clamp(src, minData, maxData);
    int4 dst = convert_int4_rte(result * outputScale + outputZP);
    write_imagei(output, coord, dst);
}

__kernel void clip_I32toI32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                     float minData,
                     float maxData,
                     float inputScale,
                     float inputTail,
                     float outputScale,
                     float outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = convert_float4(read_imagei(input, coord)) * inputScale + inputTail;
    float4 result = clamp(src, minData, maxData);
    int4 dst = convert_int4_rte(result * outputScale + outputZP);
    write_imagei(output, coord, dst);
}

__kernel void clip_I32toF32
    (
    __read_only  image2d_array_t input,
    __write_only image2d_array_t output,
                           float minData,
                           float maxData,
                           float inputScale,
                           float inputTail,
                           float outputScale,
                           float outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = convert_float4(read_imagei(input, coord)) * inputScale + inputTail;
    float4 dst = clamp(src, minData, maxData);
    write_imagef(output, coord, dst);
}

__kernel void clip_I32toF32_2D
    (
    __read_only  image2d_t input,
    __write_only image2d_t output,
                     float minData,
                     float maxData,
                     float inputScale,
                     float inputTail,
                     float outputScale,
                     float outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = convert_float4(read_imagei(input, coord)) * inputScale + inputTail;
    float4 dst = clamp(src, minData, maxData);
    write_imagef(output, coord, dst);
}
