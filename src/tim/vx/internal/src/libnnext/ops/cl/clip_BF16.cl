#pragma OPENCL EXTENSION CL_VIV_asm : enable

__kernel void clip_BF16toBF16
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
    uint4 src0 = read_imageui(input, coord);
    src0 = src0 << 16;
    float4 src;
    _viv_asm(COPY, src, src0, 16);
    float4 dst0 = clamp(src, minData, maxData);
    uint4 dst;
    _viv_asm(COPY, dst, dst0, 16);
    dst = dst >> 16;
    write_imageui(output, coord, dst);
}

__kernel void clip_BF16toBF16_2D
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
    uint4 src0 = read_imageui(input, coord);
    src0 = src0 << 16;
    float4 src;
    _viv_asm(COPY, src, src0, 16);
    float4 dst0 = clamp(src, minData, maxData);
    uint4 dst;
    _viv_asm(COPY, dst, dst0, 16);
    dst = dst >> 16;
    write_imageui(output, coord, dst);
}
