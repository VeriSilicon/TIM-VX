
__kernel void cumsum_array_F32toF32_axis1(
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

    float sum = (float)(0);
    Tensor img1 = create_tensor_from_image2d_array(input, 4);
    Tensor img2 = create_tensor_from_image2d_array(output, 4);
    uchar* input_ptr = get_tensor_ptr_from_coord(img1, coord);
    uchar* output_ptr = get_tensor_ptr_from_coord(img2, coord);
    __global float* in_ptr = (__global float*)input_ptr;
    __global float* out_ptr = (__global float*)output_ptr;
    if(exclusive && rev)
    {
        coord_out.y = height - 1;
        output_ptr = get_tensor_ptr_from_coord(img2, coord_out);
        out_ptr = (__global float*)output_ptr;
        out_ptr[0] = sum;
        for(coord.y = height - 1; coord.y > 0; coord.y--)
        {
            input_ptr = get_tensor_ptr_from_coord(img1, coord);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            coord_out.y--;
            sum += data;

            output_ptr = get_tensor_ptr_from_coord(img2, coord_out);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else if(exclusive)
    {
        coord_out.y = 0;
        output_ptr = get_tensor_ptr_from_coord(img2, coord_out);
        out_ptr = (__global float*)output_ptr;
        out_ptr[0] = sum;
        for(coord.y = 0; coord.y < height - 1; coord.y++)
        {
            input_ptr = get_tensor_ptr_from_coord(img1, coord);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            coord_out.y++;
            sum += data;

            output_ptr = get_tensor_ptr_from_coord(img2, coord_out);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else if(rev)
    {
        for(coord.y = height - 1; coord.y >= 0; coord.y--)
        {
            input_ptr = get_tensor_ptr_from_coord(img1, coord);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            sum += data;

            output_ptr = get_tensor_ptr_from_coord(img2, coord);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
    else
    {
        for(coord.y = 0; coord.y < height; coord.y++)
        {
            input_ptr = get_tensor_ptr_from_coord(img1, coord);
            in_ptr = (__global float*)input_ptr;
            float data = in_ptr[0];
            sum += data;

            output_ptr = get_tensor_ptr_from_coord(img2, coord);
            out_ptr = (__global float*)output_ptr;
            out_ptr[0] = sum;
        }
    }
}

#define CUMSUM_ARRAY_toU8_AXIS1_SH(name, src_type) \
__kernel void cumsum_array_##name##toU8_axis1( \
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
    uint dst = (uint4)(0); \
    int tmp_zp = convert_int_rte(output_zp); \
    dst = convert_uint_sat(tmp_zp); \
 \
    float cnt = 0; \
 \
    Tensor img1 = create_tensor_from_image2d_array(input, 4); \
    Tensor img2 = create_tensor_from_image2d_array(output, 4); \
    uchar* input_ptr = get_tensor_ptr_from_coord(img1, coord); \
    uchar* output_ptr = get_tensor_ptr_from_coord(img2, coord); \
    __global src_type* in_ptr = (__global src_type*)input_ptr; \
    __global uint* out_ptr = (__global uint*)output_ptr; \
    if(exclusive && rev) \
    { \
        coord_out.y = height - 1; \
        output_ptr = get_tensor_ptr_from_coord(img2, coord_out); \
        out_ptr = (__global uint*)output_ptr; \
        out_ptr[0] = dst; \
 \
        for(coord.y = height - 1; coord.y > 0; coord.y--) \
        { \
            input_ptr = get_tensor_ptr_from_coord(img1, coord); \
            in_ptr = (__global src_type*)input_ptr; \
            src_type data = in_ptr[0]; \
            cnt += 1.0f; \
            coord_out.y--; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum * in_out_scale + tmpAlpha; \
 \
            dst = (uint)convert_int_rte(tmpSum); \
            output_ptr = get_tensor_ptr_from_coord(img2, coord_out); \
            out_ptr = (__global uint*)output_ptr; \
            out_ptr[0] = dst; \
        } \
    } \
    else if(exclusive) \
    { \
        coord_out.y = 0; \
        output_ptr = get_tensor_ptr_from_coord(img2, coord_out); \
        out_ptr = (__global uint*)output_ptr; \
        out_ptr[0] = dst; \
        for(coord.y = 0; coord.y < height - 1; coord.y++) \
        { \
            input_ptr = get_tensor_ptr_from_coord(img1, coord); \
            in_ptr = (__global src_type*)input_ptr; \
            src_type data = in_ptr[0]; \
            cnt += 1.0f; \
            coord_out.y++; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum * in_out_scale + tmpAlpha; \
 \
            dst = (uint)convert_int_rte(tmpSum); \
            output_ptr = get_tensor_ptr_from_coord(img2, coord_out); \
            out_ptr = (__global uint*)output_ptr; \
            out_ptr[0] = dst; \
        } \
    } \
    else if(rev) \
    { \
        for(coord.y = height - 1; coord.y >= 0; coord.y--) \
        { \
            input_ptr = get_tensor_ptr_from_coord(img1, coord); \
            in_ptr = (__global src_type*)input_ptr; \
            src_type data = in_ptr[0]; \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum * in_out_scale + tmpAlpha; \
 \
            dst = (uint)convert_int_rte(tmpSum); \
            output_ptr = get_tensor_ptr_from_coord(img2, coord); \
            out_ptr = (__global uint*)output_ptr; \
            out_ptr[0] = dst; \
        } \
    } \
    else \
    { \
        for(coord.y = 0; coord.y < height; coord.y++) \
        { \
            input_ptr = get_tensor_ptr_from_coord(img1, coord); \
            in_ptr = (__global src_type*)input_ptr; \
            src_type data = in_ptr[0]; \
            cnt += 1.0f; \
            sum += data; \
 \
            float tmpAlpha = cnt * in_out_zp_scale + output_zp; \
            float tmpSum = sum * in_out_scale + tmpAlpha; \
 \
            dst = (uint)convert_int_rte(tmpSum); \
            output_ptr = get_tensor_ptr_from_coord(img2, coord); \
            out_ptr = (__global uint*)output_ptr; \
            out_ptr[0] = dst; \
        } \
    } \
}
CUMSUM_ARRAY_toU8_AXIS1_SH(U8,uint)
CUMSUM_ARRAY_toU8_AXIS1_SH(F32,float)
