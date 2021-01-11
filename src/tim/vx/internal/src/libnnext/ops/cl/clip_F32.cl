__kernel void clip_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  minData,
                           float  maxData)
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = read_imagef(input, coord);
    float4 dst = src > minData ? src : minData;
    dst = dst < maxData ? dst : maxData;
    write_imagef(output, coord, dst);
}

__kernel void clip_F32toF32_2D(
    __read_only  image2d_t  input,
    __write_only image2d_t  output,
                     float  minData,
                     float  maxData)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = read_imagef(input, coord);
    float4 dst = src > minData ? src : minData;
    dst = dst < maxData ? dst : maxData;
    write_imagef(output, coord, dst);
}

__kernel void clip_F32toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  minData,
                           float  maxData,
                           float inputScale,
                           float inputTail,
                           float outputScale,
                           float outputZP
                           )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = read_imagef(input, coord);
    float4 result = src > minData ? src : minData;
    result = result < maxData ? result : maxData;
    uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
    write_imageui(output, coord, dst);
}

__kernel void clip_F32toU8_2D(
    __read_only  image2d_t  input,
    __write_only image2d_t  output,
                     float  minData,
                     float  maxData,
                     float inputScale,
                     float inputTail,
                     float outputScale,
                     float outputZP
                     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = read_imagef(input, coord);
    float4 result = src > minData ? src : minData;
    result = result < maxData ? result : maxData;
    uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
    write_imageui(output, coord, dst);
}

