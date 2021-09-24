__kernel void moments_axis012_U8toF32(
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari,
    int axis, int axis_num, int input_zp, float input_scale,
    int width, int height, int chn, float dimRatio
    )
{
    int gidx = get_global_id(0);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, 0, 0);
    uint4 data;
    float sum = 0, sqr = 0;
    float e2InScale = input_scale * input_scale;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(coord.z = 0; coord.z < chn; coord.z++)
    {
        for(coord.x = gidx; coord.x < width; coord.x += 16)
        {
            int tmpSum = 0, tmpSqr = 0;
            for(coord.y = 0; coord.y < height;)
            {
                data = read_imageui(input, coord);
                coord.y++;
                tmpSum += data.x;
                tmpSqr += data.x * data.x;
            }
            sqr += (tmpSqr - 2 * input_zp * tmpSum + height * input_zp * input_zp) * e2InScale;
            sum += (tmpSum - height * input_zp) * input_scale;
        }
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord_out = (int2)(0, 0);
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

#define MOMENTS_AXIS012_F(src0_type_name) \
__kernel void moments_axis012_##src0_type_name##to##src0_type_name( \
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari, \
    int axis, int axis_num, int input_zp, float input_scale, \
    int width, int height, int chn, float dimRatio \
    ) \
{ \
    int gidx = get_global_id(0); \
    int lidx = get_local_id(0); \
 \
    int4 coord = (int4)(gidx, 0, 0, 0); \
    float4 data; \
    float sum = 0, sqr = 0; \
 \
    __local float lcl_sum[16]; \
    __local float lcl_sqr[16]; \
 \
    for(coord.z = 0; coord.z < chn; coord.z++) \
    { \
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
    } \
    lcl_sum[lidx] = sum; \
    lcl_sqr[lidx] = sqr; \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    int2 coord_out = (int2)(0, 0); \
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
MOMENTS_AXIS012_F(F32)

__kernel void moments_axis012_I32toF32(
    image2d_array_t   input, image2d_t  output_mean, image2d_t  output_vari,
    int axis, int axis_num, int input_zp, float input_scale,
    int width, int height, int chn, float dimRatio
    )
{
    int gidx = get_global_id(0);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, 0, 0);
    int4 data;
    float sum = 0, sqr = 0;
    float e2InScale = input_scale * input_scale;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    for(coord.z = 0; coord.z < chn; coord.z++)
    {
        for(coord.x = gidx; coord.x < width; coord.x += 16)
        {
            int tmpSum = 0, tmpSqr = 0;
            for(coord.y = 0; coord.y < height;)
            {
                data = read_imagei(input, coord);
                coord.y++;
                tmpSum = tmpSum + data.x;
                tmpSqr = tmpSqr + data.x * data.x;
            }
            sqr += (tmpSqr - 2 * input_zp * tmpSum + height * input_zp * input_zp) * e2InScale;
            sum += (tmpSum - height * input_zp) * input_scale;
        }
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord_out = (int2)(0, 0);
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
