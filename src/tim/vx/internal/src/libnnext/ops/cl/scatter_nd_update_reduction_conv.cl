__kernel void scatter_nd_update_reduction_conv_F16(
    __read_only image2d_t  temp_buf_float,
    __read_only image2d_t  link_buf,
    image2d_t  output,
    int length, int res, float output_scale, float output_zp)
{
    int gidx = get_global_id(0);
    Image img1 = create_image_from_image2d(temp_buf_float, 4);
    Image img2 = create_image_from_image2d(output, 2);
    __global float* input_ptr = (__global float*)img1.ptr;
    __global short* output_ptr = (__global short*)img2.ptr;
    if(length > 0)
    {
        int offset = gidx * 8;
        float4 src0 = vload4(0, input_ptr + offset);
        float4 src1 = vload4(1, input_ptr + offset);
        half4 data0, data1;
        _viv_asm(CONV, data0, src0);
        _viv_asm(CONV, data1, src1);
        short4 dst0, dst1;
        _viv_asm(COPY, dst0, data0, 16);
        _viv_asm(COPY, dst1, data1, 16);
        vstore4(dst0, 0, output_ptr + offset);
        vstore4(dst1, 1, output_ptr + offset);
    }
    for(int i = gidx; i < res; i += get_global_size(0))
    {
        float src = input_ptr[length + i];
        half data;
        _viv_asm(CONV, data, src);
        short dst;
        _viv_asm(COPY, dst, data, 4);
        output_ptr[length + i] = dst;
    }
}

#define SCATTER_ND_UPDATE_CONV(src0_type, ptr_type, element_size, ptr_type1, conv_func) \
__kernel void scatter_nd_update_reduction_conv_##src0_type( \
    __read_only image2d_t  temp_buf_float, \
    __read_only image2d_t  link_buf, \
    image2d_t  output, \
    int length, int res, float output_scale, float output_zp) \
{ \
    int gidx = get_global_id(0); \
    Image img1 = create_image_from_image2d(temp_buf_float, 4); \
    Image img2 = create_image_from_image2d(output, element_size); \
    __global float* input_ptr = (__global float*)img1.ptr; \
    __global ptr_type1* output_ptr = (__global ptr_type1*)img2.ptr; \
    if(length > 0) \
    { \
        int offset = gidx * 8; \
        float4 src0 = vload4(0, input_ptr + offset); \
        float4 src1 = vload4(1, input_ptr + offset); \
        int4 data0 = convert_int4_rte(src0 * output_scale + output_zp); \
        int4 data1 = convert_int4_rte(src1 * output_scale + output_zp); \
        ptr_type dst0, dst1; \
        _viv_asm(CONV, dst0, data0); \
        _viv_asm(CONV, dst1, data1); \
        vstore4(dst0, 0, output_ptr + offset); \
        vstore4(dst1, 1, output_ptr + offset); \
    } \
    for(int i = gidx; i < res; i += get_global_size(0)) \
    { \
        float src = input_ptr[length + i]; \
        int data = convert_int_rte(src * output_scale + output_zp); \
        output_ptr[length + i] = conv_func(data); \
    } \
}
SCATTER_ND_UPDATE_CONV(U8,  uchar4, 1, uchar, convert_uchar)
SCATTER_ND_UPDATE_CONV(I8,  char4,  1, char,  convert_char)
SCATTER_ND_UPDATE_CONV(I16, short4, 2, short, convert_short)
SCATTER_ND_UPDATE_CONV(F32, float4, 4, float, convert_float)
