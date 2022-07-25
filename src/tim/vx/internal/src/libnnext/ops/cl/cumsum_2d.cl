
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

__kernel void cumsum_U8toU8_axis1_2D(
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

    uint4 sum = (uint4)(0);
    uint4 dst = (uint4)(0);

    float cnt = 0;

    if(exclusive && rev)
    {
        coord.w = height - 1;
        write_imageui(output, coord.zw, sum);
        for(coord.y = height - 1; coord.y > 0; coord.y--)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            coord.w--;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.zw, dst);
        }
    }
    else if(exclusive)
    {
        write_imageui(output, coord.zw, sum);
        for(coord.y = 0; coord.y < height - 1; coord.y++)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            coord.w++;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.zw, dst);
        }
    }
    else if(rev)
    {
        for(coord.y = height - 1; coord.y >= 0; coord.y--)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.xy, dst);
        }
    }
    else
    {
        for(coord.y = 0; coord.y < height; coord.y++)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.xy, dst);
        }
    }
}

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

__kernel void cumsum_U8toU8_axis0_2D(
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

    uint4 sum = (uint4)(0);
    uint4 dst = (uint4)(0);

    float cnt = 0.0f;

    if(exclusive && rev)
    {
        coord.x = width - 1;
        coord.z = coord.x;
        write_imageui(output, coord.zw, sum);
        for(; coord.x > 0; coord.x--)
        {
            uint4 data = read_imageui(input, coord.xy);
            coord.z--;
            cnt += 1.0;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.zw, dst);
        }
    }
    else if(exclusive)
    {
        coord.z = 0;
        write_imageui(output, coord.zw, sum);
        for(coord.x = 0; coord.x < width - 1; coord.x++)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            coord.z++;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.zw, dst);
        }
    }
    else if(rev)
    {
        for(coord.x = width - 1; coord.x >= 0; coord.x--)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.xy, dst);
        }
    }
    else
    {
        for(coord.x = 0; coord.x < width; coord.x++)
        {
            uint4 data = read_imageui(input, coord.xy);
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            write_imageui(output, coord.xy, dst);
        }
    }
}
