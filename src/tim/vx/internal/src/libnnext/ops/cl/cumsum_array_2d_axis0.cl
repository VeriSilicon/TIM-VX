
__kernel void cumsum_array_F32toF32_axis0_2D(
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

    float sum = (float)(0);
    Image img1 = create_image_from_image2d(input, 4);
    Image img2 = create_image_from_image2d(output, 4);
    uchar* input_ptr = get_image_ptr_from_coord(img1, coord);
    uchar* output_ptr = get_image_ptr_from_coord(img2, coord);
    __global float* in_ptr = (__global float*)input_ptr;
    __global float* out_ptr = (__global float*)output_ptr;
    if(exclusive && rev)
    {
        coord.x = width - 1;
        coord.z = coord.x;
        output_ptr = get_image_ptr_from_coord(img2, coord.zw);
        out_ptr = (__global float*)output_ptr;
        out_ptr[0] = sum;

        for(; coord.x > 0; coord.x--)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            coord.z--;
            sum += data;

            output_ptr = get_image_ptr_from_coord(img2, coord.zw);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else if(exclusive)
    {
        coord.z = 0;
        output_ptr = get_image_ptr_from_coord(img2, coord.zw);
        out_ptr = (__global float*)output_ptr;
        out_ptr[0] = sum;
        for(coord.x = 0; coord.x < width - 1; coord.x++)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            coord.z++;
            sum += data;

            output_ptr = get_image_ptr_from_coord(img2, coord.zw);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else if(rev)
    {
        for(coord.x = width - 1; coord.x >= 0; coord.x--)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            sum += data;

            output_ptr = get_image_ptr_from_coord(img2, coord.xy);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else
    {
        for(coord.x = 0; coord.x < width; coord.x++)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            sum += data;

            output_ptr = get_image_ptr_from_coord(img2, coord.xy);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
}

__kernel void cumsum_array_U8toU8_axis0_2D(
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

    uint sum = (uint)(0);
    uint dst = (uint)(0);

    int tmp_zp = convert_int_rte(output_zp);
    dst.x = convert_uint_sat(tmp_zp);

    float cnt = 0.0f;

    Image img1 = create_image_from_image2d(input, 4);
    Image img2 = create_image_from_image2d(output, 4);
    uchar* input_ptr = get_image_ptr_from_coord(img1, coord);
    uchar* output_ptr = get_image_ptr_from_coord(img2, coord);
    __global uint* in_ptr = (__global uint*)input_ptr;
    __global uint* out_ptr = (__global uint*)output_ptr;
    if(exclusive && rev)
    {
        coord.x = width - 1;
        coord.z = coord.x;
        output_ptr = get_image_ptr_from_coord(img2, coord.zw);
        out_ptr = (__global float*)output_ptr;
        out_ptr[0] = dst;
        for(; coord.x > 0; coord.x--)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global uint*)input_ptr;
            uint data = in_ptr[0];
            coord.z--;
            cnt += 1.0;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.zw);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else if(exclusive)
    {
        coord.z = 0;
        output_ptr = get_image_ptr_from_coord(img2, coord.zw);
        out_ptr = (__global float*)output_ptr;
        out_ptr[0] = dst;
        for(coord.x = 0; coord.x < width - 1; coord.x++)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global uint*)input_ptr;
            uint data = in_ptr[0];
            cnt += 1.0f;
            coord.z++;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.zw);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = dst;
        }
    }
    else if(rev)
    {
        for(coord.x = width - 1; coord.x >= 0; coord.x--)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global uint*)input_ptr;
            uint data = in_ptr[0];
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.xy);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = dst;
        }
    }
    else
    {
        for(coord.x = 0; coord.x < width; coord.x++)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global uint*)input_ptr;
            uint data = in_ptr[0];
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.xy);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = dst;
        }
    }
}

__kernel void cumsum_array_F32toU8_axis0_2D(
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
    uint4 dst = (uint4)(0);
    int tmp_zp = convert_int_rte(output_zp);
    dst.x = convert_uint_sat(tmp_zp);

    float cnt = 0.0f;
    Image img1 = create_image_from_image2d(input, 4);
    Image img2 = create_image_from_image2d(output, 4);
    uchar* input_ptr = get_image_ptr_from_coord(img1, coord);
    uchar* output_ptr = get_image_ptr_from_coord(img2, coord);
    __global float* in_ptr = (__global float*)input_ptr;
    __global uint* out_ptr = (__global uint*)output_ptr;
    if(exclusive && rev)
    {
        coord.x = width - 1;
        coord.z = coord.x;
        output_ptr = get_image_ptr_from_coord(img2, coord.zw);
        out_ptr = (__global uint*)output_ptr;
        out_ptr[0] = dst;
        for(; coord.x > 0; coord.x--)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            coord.z--;
            cnt += 1.0;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.zw);
            out_ptr = (__global uint*)output_ptr;
            out_ptr[0] = dst;
        }
    }
    else if(exclusive)
    {
        coord.z = 0;
        output_ptr = get_image_ptr_from_coord(img2, coord.zw);
        out_ptr = (__global uint*)output_ptr;
        out_ptr[0] = dst;
        for(coord.x = 0; coord.x < width - 1; coord.x++)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            cnt += 1.0f;
            coord.z++;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.zw);
            out_ptr = (__global uint*)output_ptr;
            out_ptr[0] = dst;
        }
    }
    else if(rev)
    {
        for(coord.x = width - 1; coord.x >= 0; coord.x--)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum * in_out_scale + tmpAlpha;

            dst = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.xy);
            out_ptr = (__global uint*)output_ptr;
            out_ptr[0] = dst;
        }
    }
    else
    {
        for(coord.x = 0; coord.x < width; coord.x++)
        {
            input_ptr = get_image_ptr_from_coord(img1, coord.xy);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            cnt += 1.0f;
            sum += data;

            float tmpAlpha = cnt * in_out_zp_scale + output_zp;
            float tmpSum = sum.x * in_out_scale + tmpAlpha;

            dst.x = (uint)convert_int_rte(tmpSum);
            output_ptr = get_image_ptr_from_coord(img2, coord.xy);
            out_ptr = (__global uint*)output_ptr;
            out_ptr[0] = dst;
        }
    }
}
