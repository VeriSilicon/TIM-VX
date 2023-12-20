__kernel void cumsum_F32toF32_axis2(
    __read_only image2d_array_t  input,
    __write_only image2d_array_t  output,
    int axis,
    int exclusive,
    int rev,
    int width,
    int height,
    int channel,
    int input_zp,
    float in_out_scale,
    float in_out_zp_scale,
    float output_zp
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_out = coord;

    float4 sum = (float4)(0);

    if(exclusive && rev)
    {
        coord_out.z = channel - 1;
        write_imagef(output, coord_out, sum);

        for(coord.z = channel - 1; coord.z > 0; coord.z--)
        {
            float4 data = read_imagef(input, coord);
            coord_out.z--;
            sum += data;

            write_imagef(output, coord_out, sum);
        }
    }
    else if(exclusive)
    {
        coord_out.z = 0;
        write_imagef(output, coord_out, sum);
        for(coord.z = 0; coord.z < channel - 1; coord.z++)
        {
            float4 data = read_imagef(input, coord);
            coord_out.z++;
            sum += data;

            write_imagef(output, coord_out, sum);
        }
    }
    else if(rev)
    {
        for(coord.z = channel - 1; coord.z >= 0; coord.z--)
        {
            float4 data = read_imagef(input, coord);
            sum += data;

            write_imagef(output, coord, sum);
        }
    }
    else
    {
        for(coord.z = 0; coord.z < channel; coord.z++)
        {
            float4 data = read_imagef(input, coord);
            sum += data;

            write_imagef(output, coord, sum);
        }
    }
}

#define CUMSUM_toU8_AXIS2_SH(name, src_type, read_image_type) \
__kernel void cumsum_##name##toU8_axis2( \
    __read_only image2d_array_t  input, \
    __write_only image2d_array_t  output, \
    int axis, \
    int exclusive, \
    int rev, \
    int width, \
    int height, \
    int channel, \
    int input_zp, \
    float in_out_scale, \
    float in_out_zp_scale, \
    float output_zp \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int4 coord_out = coord; \
 \
    src_type sum = (src_type)(0); \
    uint4 dst = (uint4)(0); \
    int tmp_zp = convert_int_rte(output_zp); \
    dst.x = convert_uint_sat(tmp_zp); \
 \
    float cnt = 0.0f; \
 \
    if(exclusive && rev) \
    { \
        coord_out.z = channel - 1; \
        write_imageui(output, coord_out, dst); \
        for(coord.z = channel - 1; coord.z > 0; coord.z--) \
        { \
            src_type data = read_image_type(input, coord); \
            coord_out.z--; \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord_out, dst); \
        } \
    } \
    else if(exclusive) \
    { \
        coord_out.z = 0; \
        write_imageui(output, coord_out, dst); \
        for(coord.z = 0; coord.z < channel - 1; coord.z++) \
        { \
            src_type data = read_image_type(input, coord); \
            coord_out.z++; \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord_out, dst); \
        } \
    } \
    else if(rev) \
    { \
        for(coord.z = channel - 1; coord.z >= 0; coord.z--) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord, dst); \
        } \
    } \
    else \
    { \
        for(coord.z = 0; coord.z < channel; coord.z++) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord, dst); \
        } \
    } \
}
CUMSUM_toU8_AXIS2_SH(U8,uint4,read_imageui)
CUMSUM_toU8_AXIS2_SH(F32,float4,read_imagef)



__kernel void cumsum_F32toF32_axis1(
    __read_only image2d_array_t  input,
    __write_only image2d_array_t  output,
    int axis,
    int exclusive,
    int rev,
    int width,
    int height,
    int channel,
    int input_zp,
    float in_out_scale,
    float in_out_zp_scale,
    float output_zp
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_out = coord;

    float4 sum = (float4)(0);

    if(exclusive && rev)
    {
        coord_out.y = height - 1;
        write_imagef(output, coord_out, sum);
        for(coord.y = height - 1; coord.y > 0; coord.y--)
        {
            float4 data = read_imagef(input, coord);
            coord_out.y--;
            sum += data;

            write_imagef(output, coord_out, sum);
        }
    }
    else if(exclusive)
    {
        coord_out.y = 0;
        write_imagef(output, coord_out, sum);
        for(coord.y = 0; coord.y < height - 1; coord.y++)
        {
            float4 data = read_imagef(input, coord);
            coord_out.y++;
            sum += data;

            write_imagef(output, coord_out, sum);
        }
    }
    else if(rev)
    {
        for(coord.y = height - 1; coord.y >= 0; coord.y--)
        {
            float4 data = read_imagef(input, coord);
            sum += data;

            write_imagef(output, coord, sum);
        }
    }
    else
    {
        for(coord.y = 0; coord.y < height; coord.y++)
        {
            float4 data = read_imagef(input, coord);
            sum += data;

            write_imagef(output, coord, sum);
        }
    }
}

#define CUMSUM_toU8_AXIS1_SH(name, src_type, read_image_type) \
__kernel void cumsum_##name##toU8_axis1( \
    __read_only image2d_array_t  input, \
    __write_only image2d_array_t  output, \
    int axis, \
    int exclusive, \
    int rev, \
    int width, \
    int height, \
    int channel, \
    int input_zp, \
    float in_out_scale, \
    float in_out_zp_scale, \
    float output_zp \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int4 coord_out = coord; \
 \
    src_type sum = (src_type)(0); \
    uint4 dst = (uint4)(0); \
    int tmp_zp = convert_int_rte(output_zp); \
    dst.x = convert_uint_sat(tmp_zp); \
 \
    float cnt = 0; \
 \
    if(exclusive && rev) \
    { \
        coord_out.y = height - 1; \
        write_imageui(output, coord_out, dst); \
 \
        for(coord.y = height - 1; coord.y > 0; coord.y--) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            coord_out.y--; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord_out, dst); \
        } \
    } \
    else if(exclusive) \
    { \
        coord_out.y = 0; \
        write_imageui(output, coord_out, dst); \
        for(coord.y = 0; coord.y < height - 1; coord.y++) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            coord_out.y++; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord_out, dst); \
        } \
    } \
    else if(rev) \
    { \
        for(coord.y = height - 1; coord.y >= 0; coord.y--) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord, dst); \
        } \
    } \
    else \
    { \
        for(coord.y = 0; coord.y < height; coord.y++) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord, dst); \
        } \
    } \
}
CUMSUM_toU8_AXIS1_SH(U8,uint4,read_imageui)
CUMSUM_toU8_AXIS1_SH(F32,float4,read_imagef)


__kernel void cumsum_F32toF32_axis0(
    __read_only image2d_array_t  input,
    __write_only image2d_array_t  output,
    int axis,
    int exclusive,
    int rev,
    int width,
    int height,
    int channel,
    int input_zp,
    float in_out_scale,
    float in_out_zp_scale,
    float output_zp
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_out = coord;

    float4 sum = (float4)(0);

    if(exclusive && rev)
    {
        coord_out.x = width - 1;
        write_imagef(output, coord_out, sum);
        for(coord.x = width - 1; coord.x > 0; coord.x--)
        {
            float4 data = read_imagef(input, coord);
            coord_out.x--;
            sum += data;

            write_imagef(output, coord_out, sum);
        }
    }
    else if(exclusive)
    {
        coord_out.x = 0;
        write_imagef(output, coord_out, sum);
        for(coord.x = 0; coord.x < width - 1; coord.x++)
        {
            float4 data = read_imagef(input, coord);
            coord_out.x++;
            sum += data;

            write_imagef(output, coord_out, sum);
        }
    }
    else if(rev)
    {
        for(coord.x = width - 1; coord.x >= 0; coord.x--)
        {
            float4 data = read_imagef(input, coord);
            sum += data;

            write_imagef(output, coord, sum);
        }
    }
    else
    {
        for(coord.x = 0; coord.x < width; coord.x++)
        {
            float4 data = read_imagef(input, coord);
            sum += data;

            write_imagef(output, coord, sum);
        }
    }
}

#define CUMSUM_toU8_AXIS0_SH(name, src_type, read_image_type) \
__kernel void cumsum_##name##toU8_axis0( \
    __read_only image2d_array_t  input, \
    __write_only image2d_array_t  output, \
    int axis, \
    int exclusive, \
    int rev, \
    int width, \
    int height, \
    int channel, \
    int input_zp, \
    float in_out_scale, \
    float in_out_zp_scale, \
    float output_zp \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int4 coord_out = coord; \
 \
    src_type sum = (src_type)(0); \
    uint4 dst = (uint4)(0); \
    int tmp_zp = convert_int_rte(output_zp); \
    dst.x = convert_uint_sat(tmp_zp); \
 \
    float cnt = 0; \
 \
    if(exclusive && rev) \
    { \
        coord_out.x = width - 1; \
        write_imageui(output, coord_out, dst); \
        for(coord.x = width - 1; coord.x > 0; coord.x--) \
        { \
            src_type data = read_image_type(input, coord); \
            coord_out.x--; \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord_out, dst); \
        } \
    } \
    else if(exclusive) \
    { \
        coord_out.x = 0; \
        write_imageui(output, coord_out, dst); \
        for(coord.x = 0; coord.x < width - 1; coord.x++) \
        { \
            src_type data = read_image_type(input, coord); \
            coord_out.x++; \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord_out, dst); \
        } \
    } \
    else if(rev) \
    { \
        for(coord.x = width - 1; coord.x >= 0; coord.x--) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord, dst); \
        } \
    } \
    else \
    { \
        for(coord.x = 0; coord.x < width; coord.x++) \
        { \
            src_type data = read_image_type(input, coord); \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum.x * in_out_scale + tmpAlpha; \
 \
            dst.x = (uint)convert_int_rte(tmpSum); \
            write_imageui(output, coord, dst); \
        } \
    } \
}
CUMSUM_toU8_AXIS0_SH(U8,uint4,read_imageui)
CUMSUM_toU8_AXIS0_SH(F32,float4,read_imagef)
