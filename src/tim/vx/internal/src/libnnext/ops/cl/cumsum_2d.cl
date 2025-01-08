
__kernel void cumsum_F32toF32_axis1_2D(
    __read_only image2d_t  input,
    __write_only image2d_t  output,
    int axis,
    int exclusive,
    int rev,
    int width,
    int height,
    int chn,
    int input_zp,
    float in_out_scale,
    float in_out_zp_scale,
    float output_zp
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    float4 sum = (float4)(0);

    if(exclusive && rev)
    {
        coord.w = height - 1;
        write_imagef(output, coord.zw, sum);
        for(coord.y = height - 1; coord.y > 0; coord.y--)
        {
            float4 data = read_imagef(input, coord.xy);
            coord.w--;
            sum += data;

            write_imagef(output, coord.zw, sum);
        }
    }
    else if(exclusive)
    {
        write_imagef(output, coord.zw, sum);
        for(coord.y = 0; coord.y < height - 1; coord.y++)
        {
            float4 data = read_imagef(input, coord.xy);
            coord.w++;
            sum += data;

            write_imagef(output, coord.zw, sum);
        }
    }
    else if(rev)
    {
        for(coord.y = height - 1; coord.y >= 0; coord.y--)
        {
            float4 data = read_imagef(input, coord.xy);
            sum += data;

            write_imagef(output, coord.xy, sum);
        }
    }
    else
    {
        for(coord.y = 0; coord.y < height; coord.y++)
        {
            float4 data = read_imagef(input, coord.xy);
            sum += data;

            write_imagef(output, coord.xy, sum);
        }
    }
}

#define CUMSUM_INT_AXIS1_2D_SH(name, src_type, image_read, dst_type, image_write, convert_dtype) \
__kernel void cumsum_##name##_axis1_2D( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
    int axis, \
    int exclusive, \
    int rev, \
    int width, \
    int height, \
    int chn, \
    int input_zp, \
    float in_out_scale, \
    float in_out_zp_scale, \
    float output_zp \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    src_type sum = (src_type)(0); \
    dst_type dst = (dst_type)(0); \
    int tmp_zp = convert_int_rte(output_zp); \
    dst.x = convert_dtype(tmp_zp); \
 \
    float cnt = 0; \
 \
    if(exclusive && rev) \
    { \
        coord.w = height - 1; \
        image_write(output, coord.zw, dst); \
        for(coord.y = height - 1; coord.y > 0; coord.y--) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            coord.w--; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.zw, dst); \
        } \
    } \
    else if(exclusive) \
    { \
        image_write(output, coord.zw, dst); \
        for(coord.y = 0; coord.y < height - 1; coord.y++) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            coord.w++; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.zw, dst); \
        } \
    } \
    else if(rev) \
    { \
        for(coord.y = height - 1; coord.y >= 0; coord.y--) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.xy, dst); \
        } \
    } \
    else \
    { \
        for(coord.y = 0; coord.y < height; coord.y++) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.xy, dst); \
        } \
    } \
}
CUMSUM_INT_AXIS1_2D_SH(U8toU8,   uint4,  read_imageui, uint4, write_imageui, convert_uint_sat_rte)
CUMSUM_INT_AXIS1_2D_SH(F32toU8,  float4, read_imagef,  uint4, write_imageui, convert_uint_sat_rte)
CUMSUM_INT_AXIS1_2D_SH(I32toI32, int4,   read_imagei,  int4,  write_imagei,  convert_int_sat_rte)

__kernel void cumsum_F32toF32_axis0_2D(
    __read_only image2d_t  input,
    __write_only image2d_t  output,
    int axis,
    int exclusive,
    int rev,
    int width,
    int height,
    int chn,
    int input_zp,
    float in_out_scale,
    float in_out_zp_scale,
    float output_zp
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    float4 sum = (float4)(0);

    if(exclusive && rev)
    {
        coord.x = width - 1;
        coord.z = coord.x;
        write_imagef(output, coord.zw, sum);
        for(; coord.x > 0; coord.x--)
        {
            float4 data = read_imagef(input, coord.xy);
            coord.z--;
            sum += data;

            write_imagef(output, coord.zw, sum);
        }
    }
    else if(exclusive)
    {
        coord.z = 0;
        write_imagef(output, coord.zw, sum);
        for(coord.x = 0; coord.x < width - 1; coord.x++)
        {
            float4 data = read_imagef(input, coord.xy);
            coord.z++;
            sum += data;

            write_imagef(output, coord.zw, sum);
        }
    }
    else if(rev)
    {
        for(coord.x = width - 1; coord.x >= 0; coord.x--)
        {
            float4 data = read_imagef(input, coord.xy);
            sum += data;

            write_imagef(output, coord.xy, sum);
        }
    }
    else
    {
        for(coord.x = 0; coord.x < width; coord.x++)
        {
            float4 data = read_imagef(input, coord.xy);
            sum += data;

            write_imagef(output, coord.xy, sum);
        }
    }
}

#define CUMSUM_INT_AXIS0_2D_SH(name, src_type, image_read, dst_type, image_write, convert_dtype) \
__kernel void cumsum_##name##_axis0_2D( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
    int axis, \
    int exclusive, \
    int rev, \
    int width, \
    int height, \
    int chn, \
    int input_zp, \
    float in_out_scale, \
    float in_out_zp_scale, \
    float output_zp \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    src_type sum = (src_type)(0); \
    dst_type dst = (dst_type)(0); \
 \
    int tmp_zp = convert_int_rte(output_zp); \
    dst.x = convert_dtype(tmp_zp); \
 \
    float cnt = 0.0f; \
 \
    if(exclusive && rev) \
    { \
        coord.x = width - 1; \
        coord.z = coord.x; \
        image_write(output, coord.zw, dst); \
        for(; coord.x > 0; coord.x--) \
        { \
            src_type data = image_read(input, coord.xy); \
            coord.z--; \
            cnt += 1.0; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.zw, dst); \
        } \
    } \
    else if(exclusive) \
    { \
        coord.z = 0; \
        image_write(output, coord.zw, dst); \
        for(coord.x = 0; coord.x < width - 1; coord.x++) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            coord.z++; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.zw, dst); \
        } \
    } \
    else if(rev) \
    { \
        for(coord.x = width - 1; coord.x >= 0; coord.x--) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.xy, dst); \
        } \
    } \
    else \
    { \
        for(coord.x = 0; coord.x < width; coord.x++) \
        { \
            src_type data = image_read(input, coord.xy); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = convert_dtype(tmpSum); \
            image_write(output, coord.xy, dst); \
        } \
    } \
}
CUMSUM_INT_AXIS0_2D_SH(U8toU8,   uint4,  read_imageui, uint4, write_imageui, convert_uint_sat_rte)
CUMSUM_INT_AXIS0_2D_SH(F32toU8,  float4, read_imagef,  uint4, write_imageui, convert_uint_sat_rte)
CUMSUM_INT_AXIS0_2D_SH(I32toI32, int4,   read_imagei,  int4,  write_imagei,  convert_int_sat_rte)
