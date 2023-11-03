__kernel void moments_axis01_U8toF32(
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari,
    int axis, int axis_num, int input_zp, float input_scale,
    int width, int height, int chn, float dimRatio
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, gidz, 0);
    uint4 data;
    float sum = 0, sqr = 0;
    float e2InScale = input_scale * input_scale;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(coord.x = gidx; coord.x < width; coord.x += 16)
    {
        int tmpSum = 0, tmpSqr = 0;
        for(coord.y = 0; coord.y < height;)
        {
            data = read_imageui(input, coord);
            coord.y++;
            tmpSum = tmpSum + data.x;
            tmpSqr = tmpSqr + data.x * data.x;
        }
        sqr += (tmpSqr - 2 * input_zp * tmpSum + height * input_zp * input_zp) * e2InScale;
        sum += (tmpSum - height * input_zp) * input_scale;
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord_out = (int2)(gidz, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 mean, vari;
        mean.x = sum * dimRatio;
        vari.x = sqr * dimRatio;
        vari.x = vari.x - mean.x * mean.x;

        write_imagef(output_mean, coord_out, mean);
        write_imagef(output_vari, coord_out, vari);
    }
}

#define MOMENTS_AXIS01_F(src0_type_name) \
__kernel void moments_axis01_##src0_type_name##to##src0_type_name( \
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari, \
    int axis, int axis_num, int input_zp, float input_scale, \
    int width, int height, int chn, float dimRatio \
    ) \
{ \
    int gidx = get_global_id(0); \
    int gidz = get_global_id(1); \
    int lidx = get_local_id(0); \
 \
    int4 coord = (int4)(gidx, 0, gidz, 0); \
    float4 data; \
    float sum = 0, sqr = 0; \
 \
    __local float lcl_sum[16]; \
    __local float lcl_sqr[16]; \
 \
    for(coord.x = gidx; coord.x < width; coord.x += 16) \
    { \
        for(coord.y = 0; coord.y < height;) \
        { \
            data = read_imagef(input, coord); \
            coord.y++; \
            sum += data.x; \
            sqr += data.x * data.x; \
        } \
    } \
    lcl_sum[lidx] = sum; \
    lcl_sqr[lidx] = sqr; \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    int2 coord_out = (int2)(gidz, 0); \
    if(lidx == 0) \
    { \
        float4 one = (float4)(1, 1, 1, 1); \
        __local float4* tmp_sum = (__local float4*)lcl_sum; \
        __local float4* tmp_sqr = (__local float4*)lcl_sqr; \
 \
        sum = 0; sqr = 0; \
        for(int i = 0; i < 4; i++) \
        { \
            sum += dot(tmp_sum[i], one); \
            sqr += dot(tmp_sqr[i], one); \
        } \
 \
        float4 mean, vari; \
        mean.x = sum * dimRatio; \
        vari.x = sqr * dimRatio; \
        vari.x = vari.x - mean.x * mean.x; \
 \
        write_imagef(output_mean, coord_out, mean); \
        write_imagef(output_vari, coord_out, vari); \
    } \
}
MOMENTS_AXIS01_F(F32)

__kernel void moments_axis01_I32toF32(
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari,
    int axis, int axis_num, int input_zp, float input_scale,
    int width, int height, int chn, float dimRatio
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, gidz, 0);
    float4 data;
    float sum = 0, sqr = 0;
    float e2InScale = input_scale * input_scale;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(coord.x = gidx; coord.x < width; coord.x += 16)
    {
        float tmpSum = 0, tmpSqr = 0;
        for(coord.y = 0; coord.y < height;)
        {
            data = convert_float4(read_imagei(input, coord));
            coord.y++;

            tmpSum = tmpSum + data.x;
            tmpSqr = tmpSqr + data.x * data.x;
        }
        sqr += (tmpSqr - 2 * input_zp * tmpSum + height * input_zp * input_zp) * e2InScale;
        sum += (tmpSum - height * input_zp) * input_scale;
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord_out = (int2)(gidz, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 mean, vari;
        mean.x = sum * dimRatio;
        vari.x = sqr * dimRatio;
        vari.x = vari.x - mean.x * mean.x;
        write_imagef(output_mean, coord_out, mean);
        write_imagef(output_vari, coord_out, vari);
    }
}

__kernel void moments_axis01_BF16toF32(
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari,
    int axis, int axis_num, int input_zp, float input_scale,
    int width, int height, int chn, float dimRatio
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, gidz, 0);
    float4 data;
    float sum = 0, sqr = 0;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(coord.x = gidx; coord.x < width; coord.x += 16)
    {
        float tmpSum = 0, tmpSqr = 0;
        for(coord.y = 0; coord.y < height;)
        {
            uint4 src0 = read_imageui(input, coord);
            src0 = src0 << 16;
            _viv_asm(COPY, data, src0, 16);
            coord.y++;

            tmpSum = tmpSum + data.x;
            tmpSqr = tmpSqr + data.x * data.x;
        }
        sqr += tmpSqr;
        sum += tmpSum;
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord_out = (int2)(gidz, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 mean, vari;
        mean.x = sum * dimRatio;
        vari.x = sqr * dimRatio;
        vari.x = vari.x - mean.x * mean.x;
        write_imagef(output_mean, coord_out, mean);
        write_imagef(output_vari, coord_out, vari);
    }
}

__kernel __attribute__((reqd_work_group_size(8, 8, 1))) void moments_axis12_U8toF32(
    image2d_array_t   input, image2d_array_t  output_mean, image2d_array_t  output_vari,
    int axis, int axis_num, int input_zp, float input_scale,
    int width, int height, int chn, float dimRatio
    )
{
    int lidx = get_local_id(0);
    int lidy = get_local_id(1);
    int gidz = get_global_id(2); // width

    int4 coord = (int4)(gidz, lidx, lidy, 0);
    uint4 data;
    float sum = 0, sqr = 0;
    float e2InScale = input_scale * input_scale;

    __local uint lcl_sumSqr[128];
    __local uint lcl_sumSqr1[32];

    uint2 tmpSumSqr = 0;
    for(coord.z = lidy; coord.z < chn; coord.z += 8)
    {
        for(coord.y = lidx; coord.y < height;)
        {
            data = read_imageui(input, coord);
            coord.y += 8;
            tmpSumSqr = tmpSumSqr + (uint2)(data.x, data.x * data.x);
        }
        //sqr += (tmpSqr - 2 * input_zp * tmpSum + height * input_zp * input_zp) * e2InScale;
        //sum += (tmpSum - height * input_zp) * input_scale;
    }
    int index = lidx + lidy * 8;
    vstore2(tmpSumSqr, index, lcl_sumSqr);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(index < 16)
    {
        uint4 val0 = vload4(index, lcl_sumSqr);
        uint4 val1 = vload4(index, lcl_sumSqr + 64);
        val0 += val1;
        uint2 val2 = val0.xy + val0.zw;
        vstore2(val2, index, lcl_sumSqr1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(index == 0)
    {
        uint4 val0 = 0;
        for(int i = 0; i < 8; i++)
        {
            val0 += vload4(i, lcl_sumSqr1);
        }

        float2 tmpVal = convert_float2(val0.xy + val0.zw);
        sum = (tmpVal.x - height * chn * input_zp) * input_scale;
        sqr = (tmpVal.y - 2 * input_zp * tmpVal.x + height * chn * input_zp * input_zp) * e2InScale;
        float4 mean, vari;
        mean.x = sum * dimRatio;
        vari.x = sqr * dimRatio;
        vari.x = vari.x - mean.x * mean.x;

        write_imagef(output_mean, coord.xwww, mean);
        write_imagef(output_vari, coord.xwww, vari);
    }
}
