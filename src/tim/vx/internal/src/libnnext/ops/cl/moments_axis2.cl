__kernel void moments_axis2_U8toF32(
    __read_only image2d_array_t   input,
    __write_only image2d_t  output_mean,
    __write_only image2d_t  output_vari,
    int axis,
    int axis_num,
    int input_zp,
    float input_scale,
    int width,
    int height,
    int chn,
    float dimRatio
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int4 coord0 = (int4)(gidx, gidy, 0, 0);
    uint data;
    float sum = 0, sqr = 0;
    uint tmpSum = 0, tmpSqr = 0;
    float e2InScale = input_scale * input_scale;

    {
        for(coord0.z = 0; coord0.z < chn;)
        {
            data = read_imageui(input, coord0).x;
            coord0.z++;
            tmpSum = tmpSum + data;
            tmpSqr = tmpSqr + data * data;
        }
        sqr = as_int(tmpSqr - 2 * input_zp * tmpSum + chn * input_zp * input_zp) * e2InScale;
        sum = tmpSum * input_scale;
    }

    float4 mean, vari;
    mean.x = sum * dimRatio - input_zp * input_scale;
    vari.x = sqr * dimRatio;
    vari.x = vari.x - mean.x * mean.x;

    int2 coord_out = (int2)(gidx, gidy);
    write_imagef(output_mean, coord_out, mean);
    write_imagef(output_vari, coord_out, vari);
}

#define MOMENTS_AXIS2_F(src0_type_name) \
__kernel void moments_axis2_##src0_type_name##to##src0_type_name( \
    __read_only image2d_array_t   input, \
    __write_only image2d_t  output_mean, \
    __write_only image2d_t  output_vari, \
    int axis, \
    int axis_num, \
    int input_zp, \
    float input_scale, \
    int width, \
    int height, \
    int chn, \
    float dimRatio \
    ) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
 \
    int4 coord0 = (int4)(gidx, gidy, 0, 0); \
    float data; \
    float sum = 0, sqr = 0; \
 \
    for(coord0.z = 0; coord0.z < chn;) \
    { \
        data = read_imagef(input, coord0).x; \
        coord0.z++; \
        sum += (data); \
        sqr += (data * data); \
    } \
 \
    float4 mean, vari; \
    mean.x = sum * dimRatio; \
    vari.x = sqr * dimRatio; \
    vari.x = vari.x - mean.x * mean.x; \
 \
    int2 coord_out = (int2)(gidx, gidy); \
    write_imagef(output_mean, coord_out, mean); \
    write_imagef(output_vari, coord_out, vari); \
}
MOMENTS_AXIS2_F(F32)

__kernel void moments_axis2_I32toF32(
    __read_only image2d_array_t   input,
    __write_only image2d_t  output_mean,
    __write_only image2d_t  output_vari,
    int axis,
    int axis_num,
    int input_zp,
    float input_scale,
    int width,
    int height,
    int chn,
    float dimRatio
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int4 coord0 = (int4)(gidx, gidy, 0, 0);
    float data;
    float sum = 0, sqr = 0;

    for(coord0.z = 0; coord0.z < chn;)
    {
        data = convert_float(read_imagei(input, coord0).x - input_zp);
        coord0.z++;


        sum = sum + data;
        sqr = sqr + data * data;
    }

    float4 mean, vari;
    mean.x = sum * dimRatio * input_scale;
    vari.x = sqr * dimRatio * input_scale * input_scale;
    vari.x = vari.x - mean.x * mean.x;

    int2 coord_out = (int2)(gidx, gidy);
    write_imagef(output_mean, coord_out, mean);
    write_imagef(output_vari, coord_out, vari);
}

__kernel void moments_axis2_BF16toF32(
    __read_only image2d_array_t   input,
    __write_only image2d_t  output_mean,
    __write_only image2d_t  output_vari,
    int axis,
    int axis_num,
    int input_zp,
    float input_scale,
    int width,
    int height,
    int chn,
    float dimRatio
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int4 coord0 = (int4)(gidx, gidy, 0, 0);
    float4 data;
    float sum = 0, sqr = 0;

    for(coord0.z = 0; coord0.z < chn;)
    {
        uint4 src0 = read_imageui(input, coord0);
        src0 = src0 << 16;
        _viv_asm(COPY, data, src0, 16);
        coord0.z++;

        sum = sum + data.x;
        sqr = sqr + data.x * data.x;
    }

    float4 mean, vari;
    mean.x = sum * dimRatio;
    vari.x = sqr * dimRatio;
    vari.x = vari.x - mean.x * mean.x;

    int2 coord_out = (int2)(gidx, gidy);
    write_imagef(output_mean, coord_out, mean);
    write_imagef(output_vari, coord_out, vari);
}
