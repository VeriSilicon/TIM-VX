

__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void add_mean_std_norm_F32_F32toF32(
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
    float rsEps, float dimRatio,
    float input0Scale, float input0Tail,
    float input1Scale, float input1Tail,
    float outputScale, float outputZP,
    int width)
{
    int lidx = get_local_id(0);
    int gidx = get_global_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));
    float4 src0, src1, result;
    float pSum = 0.0f, pSqr = 0.0f;
    float sum  = 0.0f,  sqr = 0.0f;
    float input_d = 0.0f;
    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(; coord.x < width; coord.x += 16)
    {
        src0 = read_imagef(input, coord);
        src1 = read_imagef(input1, coord);
        input_d = src0.x + src1.x;
        pSum += input_d;
        pSqr += input_d * input_d;
    }
    lcl_sum[lidx] = pSum;
    lcl_sqr[lidx] = pSqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 *pLocalPtr = (float4 *)&lcl_sum[0];
    float4 one = (float4)(1, 1, 1, 1);
    float4 data0;
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sum = dot(data0, one);
    pLocalPtr = (float4 *)&lcl_sqr[0];
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sqr = dot(data0, one);
    float mean;
    mean = sum * dimRatio;
    float vari, stddev_inv, rMeanStd;
    vari = sqr*dimRatio - mean*mean;
    stddev_inv = (vari==0 ? rsEps : rsqrt(vari));
    rMeanStd = (-mean) * stddev_inv;
    for(coord.x = gidx; coord.x < width; coord.x += 16)
    {
        src0 = read_imagef(input, coord);
        src1 = read_imagef(input1, coord);
        input_d = src0.x + src1.x;
        result.x = input_d * stddev_inv + rMeanStd;
        write_imagef(output, coord, result.xxxx);
    }
}


__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void add_mean_std_norm_U8_U8toF32(
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
    float rsEps, float dimRatio,
    float input0Scale, float input0Tail,
    float input1Scale, float input1Tail,
    float outputScale, float outputZP,
    int width)
{
    int lidx = get_local_id(0);
    int gidx = get_global_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));
    float4 src0, src1, result;
    float pSum = 0.0f, pSqr = 0.0f;
    float sum  = 0.0f,  sqr = 0.0f;
    float input_d = 0.0f;
    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(; coord.x < width; coord.x += 16)
    {
        src0 = convert_float4(read_imageui(input, coord))  * input0Scale - input0Tail;
        src1 = convert_float4(read_imageui(input1, coord)) * input1Scale - input1Tail;
        input_d = src0.x + src1.x;
        pSum += input_d;
        pSqr += input_d * input_d;
    }
    lcl_sum[lidx] = pSum;
    lcl_sqr[lidx] = pSqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 *pLocalPtr = (float4 *)&lcl_sum[0];
    float4 one = (float4)(1, 1, 1, 1);
    float4 data0;
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sum = dot(data0, one);
    pLocalPtr = (float4 *)&lcl_sqr[0];
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sqr = dot(data0, one);
    float mean;
    mean = sum * dimRatio;
    float vari, stddev_inv, rMeanStd;
    vari = sqr*dimRatio - mean*mean;
    stddev_inv = (vari==0 ? rsEps : rsqrt(vari));
    rMeanStd = (-mean) * stddev_inv;
    for(coord.x = gidx; coord.x < width; coord.x += 16)
    {
        src0 = convert_float4(read_imageui(input, coord))  * input0Scale - input0Tail;
        src1 = convert_float4(read_imageui(input1, coord)) * input1Scale - input1Tail;
        input_d = src0.x + src1.x;
        result.x = input_d * stddev_inv + rMeanStd;
        write_imagef(output, coord, result.xxxx);
    }
}


__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void add_mean_std_norm_U8_U8toU8(
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
    float rsEps, float dimRatio,
    float input0Scale, float input0Tail,
    float input1Scale, float input1Tail,
    float outputScale, float outputZP,
    int width)
{
    int lidx = get_local_id(0);
    int gidx = get_global_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));
    float4 src0, src1, result = 0.0f;
    float pSum = 0.0f, pSqr = 0.0f;
    float sum  = 0.0f,  sqr = 0.0f;
    float input_d = 0.0f;
    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(; coord.x < width; coord.x += 16)
    {
        src0 = convert_float4(read_imageui(input, coord))  * input0Scale - input0Tail;
        src1 = convert_float4(read_imageui(input1, coord)) * input1Scale - input1Tail;
        input_d = src0.x + src1.x;
        pSum += input_d;
        pSqr += input_d * input_d;
    }
    lcl_sum[lidx] = pSum;
    lcl_sqr[lidx] = pSqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 *pLocalPtr = (float4 *)&lcl_sum[0];
    float4 one = (float4)(1, 1, 1, 1);
    float4 data0;
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sum = dot(data0, one);
    pLocalPtr = (float4 *)&lcl_sqr[0];
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sqr = dot(data0, one);
    float mean;
    mean = sum * dimRatio;
    float vari, stddev_inv, rMeanStd;
    vari = sqr*dimRatio - mean*mean;
    stddev_inv = (vari==0 ? rsEps : rsqrt(vari));
    rMeanStd = (-mean) * stddev_inv;
    for(coord.x = gidx; coord.x < width; coord.x += 16)
    {
        src0 = convert_float4(read_imageui(input, coord))  * input0Scale - input0Tail;
        src1 = convert_float4(read_imageui(input1, coord)) * input1Scale - input1Tail;
        input_d = src0.x + src1.x;
        result.x = input_d * stddev_inv + rMeanStd;
        uint4 dst = convert_uint4(result * outputScale + outputZP);
        write_imageui(output, coord, dst);
    }
}

