
#define MAXUNPOOL(name, read_type, read_image_type, write_type, convert_type, writeimage_type) \
__kernel void maxunpool_##name( \
    __read_only  image2d_array_t  input0, \
    __read_only  image2d_array_t  input1, \
    __write_only image2d_array_t  output, \
                 int              width_nopad, \
                 int              height_nopad, \
                 int              width_in, \
                 int              height_in, \
                 int              batch, \
                 int              pad_left, \
                 int              pad_top, \
                 float            inputScale, \
                 float            inputTail, \
                 float            outputScale, \
                 float            outputTail \
    ) \
{ \
    uint gidx = get_global_id(0); \
    uint gidy = get_global_id(1); \
    uint gidz = get_global_id(2); \
    int gidx_in, gidy_in, gidz_in; \
    int4 coord_out = (int4)(gidx, gidy, gidz, 0); \
    write_type dst = (write_type)(0); \
    float4 dst_temp = (float4)(0); \
    int i,j,k; \
    if (gidx < pad_left || gidx >= width_nopad + pad_left || \
        gidy < pad_top || gidy >= height_nopad + pad_top) \
    { \
        dst_temp.x = outputTail; \
        dst = convert_type(dst_temp); \
        writeimage_type(output, coord_out, dst); \
        return; \
    } \
    gidx_in = gidx - pad_left; \
    gidy_in = gidy - pad_top; \
    gidz_in = gidz; \
    int index = gidz_in * height_nopad * width_nopad + gidy_in * width_nopad + gidx_in; \
    for (k = 0;k < batch;k++) \
    { \
        for (j = 0;j < height_in; j++) \
        { \
            for (i = 0;i < width_in; i++) \
            { \
                int index_useful = read_imagei(input1, (int4)(i,j,k,0)).x; \
                if (index_useful == index) \
                { \
                    read_type src = read_image_type(input0, (int4)(i,j,k,0)); \
                    dst_temp = convert_float4(src) * inputScale + inputTail; \
                    dst = convert_type(dst_temp * outputScale + outputTail); \
                    writeimage_type(output, coord_out, dst); \
                    return; \
                } \
            } \
        } \
    } \
    dst_temp.x = outputTail; \
    dst = convert_type(dst_temp); \
    writeimage_type(output, coord_out, dst); \
}
MAXUNPOOL(F32toF32,float4,read_imagef,float4,convert_float4,write_imagef)
MAXUNPOOL(F32toU32,float4,read_imagef,uint4, convert_uint4, write_imageui)
MAXUNPOOL(F32toI32,float4,read_imagef,int4,  convert_int4,  write_imagei)

MAXUNPOOL(U32toU32,uint4,read_imageui,uint4, convert_uint4, write_imageui)
MAXUNPOOL(U32toF32,uint4,read_imageui,float4,convert_float4,write_imagef)
MAXUNPOOL(U32toI32,uint4,read_imageui,int4,  convert_int4,  write_imagei)

MAXUNPOOL(I32toU32,int4,read_imagei,uint4, convert_uint4, write_imageui)
MAXUNPOOL(I32toF32,int4,read_imagei,float4,convert_float4,write_imagef)
MAXUNPOOL(I32toI32,int4,read_imagei,int4,  convert_int4,  write_imagei)

__kernel void maxunpool_BF16toBF16(
    __read_only  image2d_array_t  input0,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 int              width_nopad,
                 int              height_nopad,
                 int              width_in,
                 int              height_in,
                 int              batch,
                 int              pad_left,
                 int              pad_top,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputTail
    )
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    int gidx_in, gidy_in, gidz_in;
    int4 coord_out = (int4)(gidx, gidy, gidz, 0);
    uint4 dst = (uint4)(0);
    float4 dst_temp = (float4)(0);
    int i,j,k;
    if (gidx < pad_left || gidx >= width_nopad + pad_left ||
        gidy < pad_top || gidy >= height_nopad + pad_top)
    {
        dst_temp.x = 0;
        _viv_asm(COPY, dst, dst_temp, 16);
        dst.x = dst.x >> 16;
        write_imageui(output, coord_out, dst);
        return;
    }
    gidx_in = gidx - pad_left;
    gidy_in = gidy - pad_top;
    gidz_in = gidz;
    int index = gidz_in * height_nopad * width_nopad + gidy_in * width_nopad + gidx_in;
    for (k = 0;k < batch;k++)
    {
        for (j = 0;j < height_in; j++)
        {
            for (i = 0;i < width_in; i++)
            {
                int index_useful = read_imagei(input1, (int4)(i,j,k,0)).x;
                if (index_useful == index)
                {
                    uint4 src = read_imageui(input0, (int4)(i,j,k,0));
                    write_imageui(output, coord_out, src);
                    return;
                }
            }
        }
    }
    dst_temp.x = 0;
    _viv_asm(COPY, dst, dst_temp, 16);
    dst.x = dst.x >> 16;
    write_imageui(output, coord_out, dst);
}