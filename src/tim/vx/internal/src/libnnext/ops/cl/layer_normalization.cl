
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void layer_norm_F32toF32(
    __read_only image2d_array_t   input,
    __read_only image2d_t   bias,
    __read_only image2d_t   scale,
    __write_only image2d_array_t  output,
    float eps,
    float input_zp,
    float input_scale,
    float output_zp,
    float output_scale,
    float e2InScale,
    float scale_inOut,
    float sumZpScale,
    float zp2ScaleE2,
    float sumZpScaleE2,
    int width,
    int height,
    float dim_ratio
    )
{
    int lidx = get_local_id(0);
    int4 coord = (int4)(lidx, get_global_id(1), get_global_id(2), 0);

    float4 data, dst;
    float2 sumSqr = (float2)(0);
    float scale_vari, bias_val;
    __local float2 local_sum[16];

    for(; coord.x < width;)
    {
        data = read_imagef(input, coord);
        coord.x += 16;
        sumSqr.x += data.x;
        sumSqr.y += data.x * data.x;
    }
    local_sum[lidx] = sumSqr;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx == 0)
    {
        for(int i = 1; i < 16; i++)
        {
            sumSqr += local_sum[i];
        }
        local_sum[0] = sumSqr;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sumSqr = local_sum[0] * dim_ratio;
    sumSqr.s1 = sumSqr.s1 - sumSqr.s0 * sumSqr.s0 + eps;
    sumSqr.s1 = rsqrt(sumSqr.s1);

    for(coord.x = lidx; coord.x < width;)
    {
        float4 gamma = read_imagef(scale, coord.xw);
        float4 beta  = read_imagef(bias, coord.xw);
        data = read_imagef(input, coord);

        scale_vari = gamma.s0 * sumSqr.s1;
        bias_val = (beta.s0 - scale_vari * sumSqr.s0);

        dst.x = data.x * scale_vari + bias_val;
        write_imagef(output, coord, dst);
        coord.x += 16;
    }
}

__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void layer_norm_U8toU8(
    __read_only image2d_array_t   input,
    __read_only image2d_t   bias,
    __read_only image2d_t   scale,
    __write_only image2d_array_t  output,
    float eps,
    float input_zp,
    float input_scale,
    float output_zp,
    float output_scale,
    float e2InScale,
    float scale_inOut,
    float sumZpScale,
    float zp2ScaleE2,
    float sumZpScaleE2,
    int width,
    int height,
    float dim_ratio
    )
{
    int lidx = get_local_id(0);
    int4 coord = (int4)(lidx, get_global_id(1), get_global_id(2), 0);

    uint4 data, dst;
    float2 sumSqr;
    uint tmpSum = 0, tmpSqr = 0;
    float scale_vari, bias_val;
    __local uint local_sum[1];
    __local uint local_sqr[1];

    if(lidx == 0)
    {
        local_sum[0] = 0;
        local_sqr[0] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(; coord.x < width;)
    {
        data = read_imageui(input, coord);
        coord.x+=16;
        tmpSum += data.x;
        tmpSqr += data.x * data.x;
    }
    atom_add(local_sum, tmpSum);
    atom_add(local_sqr, tmpSqr);
    barrier(CLK_LOCAL_MEM_FENCE);
    tmpSum = local_sum[0];
    tmpSqr = local_sqr[0];
    //sumSqr.x = ((float)tmpSum - width * input_zp) * input_scale;
    //sumSqr.y = ((float)tmpSqr - 2 * input_zp * (float)tmpSum + width * input_zp * input_zp) * e2InScale;
    sumSqr.x = (float)tmpSum * input_scale - sumZpScale;
    sumSqr.y = (float)tmpSqr * e2InScale - zp2ScaleE2 * (float)tmpSum + sumZpScaleE2;

    sumSqr *= dim_ratio;
    sumSqr.s1 = sumSqr.s1 - sumSqr.s0 * sumSqr.s0 + eps;
    sumSqr.s1 = rsqrt(sumSqr.s1);

    for(coord.x = lidx; coord.x < width;)
    {
        float4 gamma = read_imagef(scale, coord.xw);
        float4 beta  = read_imagef(bias, coord.xw);
        data = read_imageui(input, coord);

        scale_vari = gamma.s0 * sumSqr.s1;
        float alpha = scale_inOut * scale_vari;
        bias_val = (beta.s0 - scale_vari * sumSqr.s0) * output_scale + output_zp;

        float tmpVal = data.x - input_zp;

        float4 norm;
        norm.x = tmpVal * alpha + bias_val;
        dst = convert_uint4_rte(norm);
        write_imageui(output, coord, dst);
        coord.x+=16;
    }
}
