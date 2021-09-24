__kernel void moments_axis0_U8toF32(
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
    int gidy = get_global_id(0);
    int gidz = get_global_id(1);

    int4 coord0 = (int4)(0, gidy, gidz, 0);
    uint data;
    float sum = 0, sqr = 0;
    uint tmpSum = 0, tmpSqr = 0;
    float e2InScale = input_scale * input_scale;

    {
        for(coord0.x = 0; coord0.x < width;)
        {
            data = read_imageui(input, coord0).x;
            coord0.x++;
            tmpSum += (data);
            tmpSqr += (data * data);
        }
        sqr = convert_float(as_int(tmpSqr - 2 * input_zp * tmpSum + width * input_zp * input_zp)) * e2InScale;
        sum = convert_float(as_int(tmpSum - width * input_zp)) * input_scale;
    }
    float4 mean, vari;
    mean.x = sum * dimRatio;
    vari.x = sqr * dimRatio;
    vari.x = vari.x - mean.x * mean.x;

    int2 coord_out = (int2)(gidy, gidz);
    write_imagef(output_mean, coord_out, mean);
    write_imagef(output_vari, coord_out, vari);
}

#define MOMENTS_AXIS0_F(src0_type_name) \
__kernel void moments_axis0_##src0_type_name##to##src0_type_name( \
    __read_only image2d_array_t   input, \
    __write_only image2d_t  output_mean, \
    __write_only image2d_t  output_vari, \
    int axis, int axis_num, int input_zp, float input_scale, \
    int width, int height, int chn, float dimRatio \
    ) \
{ \
    int gidy = get_global_id(0); \
    int gidz = get_global_id(1); \
 \
    int4 coord0 = (int4)(0, gidy, gidz, 0); \
    float data; \
    float sum = 0, sqr = 0; \
 \
    for(coord0.x = 0; coord0.x < width;) \
    { \
        data = read_imagef(input, coord0).x; \
        coord0.x++; \
        sum += (data); \
        sqr += (data * data); \
    } \
 \
    float4 mean, vari; \
    mean.x = sum * dimRatio; \
    vari.x = sqr * dimRatio; \
    vari.x = vari.x - mean.x * mean.x; \
 \
    int2 coord_out = (int2)(gidy, gidz); \
    write_imagef(output_mean, coord_out, mean); \
    write_imagef(output_vari, coord_out, vari); \
}
MOMENTS_AXIS0_F(F32)

__kernel void moments_axis0_I32toF32(
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
    int gidy = get_global_id(0);
    int gidz = get_global_id(1);

    int4 coord0 = (int4)(0, gidy, gidz, 0);
    float data;
    float sum = 0, sqr = 0;

    for(coord0.x = 0; coord0.x < width;)
    {
        data = convert_float(read_imagei(input, coord0).x);
        coord0.x++;

        sum = sum + data;
        sqr = sqr + data * data;
    }

    float4 mean, vari;
    mean.x = sum * dimRatio * input_scale;
    vari.x = sqr * dimRatio * input_scale * input_scale;
    vari.x = vari.x - mean.x * mean.x;

    int2 coord_out = (int2)(gidy, gidz);
    write_imagef(output_mean, coord_out, mean);
    write_imagef(output_vari, coord_out, vari);
}