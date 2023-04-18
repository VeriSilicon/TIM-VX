#define REVERSESEQUENCE_axis2(name,src_type,readimage_type,\
                    convert_type,dst_type,writeimage_type) \
__kernel void reversesequence_##name( \
    __read_only  image2d_array_t  input0, \
    __read_only  image2d_t        input1, \
    __write_only image2d_array_t  output, \
                 float            inoutScale, \
                 float            inoutTail \
    ) \
{ \
    uint gidx = get_global_id(0); \
    uint gidy = get_global_id(1); \
    uint gidz = get_global_id(2); \
\
    int4 coord_in = (int4)(gidx, gidy, gidz, 0); \
    int4 coord_out = coord_in; \
    src_type src = readimage_type(input0, coord_in); \
    int src_index = read_imagei(input1, (int2)(gidz, 0)).x; \
    float4 src_temp = convert_float4(src); \
    dst_type dst = convert_type(src_temp * inoutScale + inoutTail); \
    if (gidy >= src_index) \
    { \
        writeimage_type(output, coord_out, dst); \
    } \
    else \
    { \
        coord_out.y = src_index - 1 - coord_out.y; \
        writeimage_type(output, coord_out, dst); \
    } \
}
REVERSESEQUENCE_axis2(F32toF32_axis2,float4,read_imagef,\
                      convert_float4,float4,write_imagef)
REVERSESEQUENCE_axis2(F32toU32_axis2,float4,read_imagef,\
                      convert_uint4, uint4, write_imageui)
REVERSESEQUENCE_axis2(F32toI32_axis2,float4,read_imagef,\
                       convert_int4,  int4,  write_imagei)
REVERSESEQUENCE_axis2(I32toF32_axis2,int4,  read_imagei,\
                       convert_float4,float4,write_imagef)
REVERSESEQUENCE_axis2(I32toU32_axis2,int4,  read_imagei,\
                     convert_uint4, uint4, write_imageui)
REVERSESEQUENCE_axis2(I32toI32_axis2,int4,  read_imagei,\
                      convert_int4,  int4,  write_imagei)
REVERSESEQUENCE_axis2(U32toF32_axis2,uint4, read_imageui,\
                      convert_float4,float4,write_imagef)
REVERSESEQUENCE_axis2(U32toU32_axis2,uint4, read_imageui,\
                     convert_uint4, uint4, write_imageui)
REVERSESEQUENCE_axis2(U32toI32_axis2,uint4, read_imageui,\
                        convert_int4,  int4,  write_imagei)

__kernel void reversesequence_BF16toBF16_axis2(
    __read_only  image2d_array_t  input0,
    __read_only  image2d_t        input1,
    __write_only image2d_array_t  output,
                 float            inoutScale,
                 float            inoutTail
    )
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);

    int4 coord_in = (int4)(gidx, gidy, gidz, 0);
    int4 coord_out = coord_in;
    uint4 src = read_imageui(input0, coord_in);
    int src_index = read_imagei(input1, (int2)(gidz, 0)).x;
    uint4 dst = src;
    if (gidy >= src_index)
    {
        write_imageui(output, coord_out, dst);
    }
    else
    {
        coord_out.y = src_index - 1 - coord_out.y;
        write_imageui(output, coord_out, dst);
    }
}


#define REVERSESEQUENCE_axis1(name,src_type,readimage_type,\
                             convert_type,dst_type,writeimage_type) \
__kernel void reversesequence_##name( \
    __read_only  image2d_array_t  input0, \
    __read_only  image2d_t        input1, \
    __write_only image2d_array_t  output, \
                 float            inoutScale, \
                 float            inoutTail \
    ) \
{ \
    uint gidx = get_global_id(0); \
    uint gidy = get_global_id(1); \
    uint gidz = get_global_id(2); \
\
    int4 coord_in = (int4)(gidx, gidy, gidz, 0); \
    int4 coord_out = coord_in; \
    src_type src = readimage_type(input0, coord_in); \
    int src_index = read_imagei(input1, (int2)(gidy, 0)).x; \
    float4 src_temp = convert_float4(src); \
    dst_type dst = convert_type(src_temp * inoutScale + inoutTail ); \
    if (gidz >= src_index) \
    { \
        writeimage_type(output, coord_out, dst); \
    } \
    else \
    { \
        coord_out.z = src_index - 1 - coord_out.z; \
        writeimage_type(output, coord_out, dst); \
    } \
}
REVERSESEQUENCE_axis1(F32toF32_axis1,float4,read_imagef,\
                     convert_float4,float4,write_imagef)
REVERSESEQUENCE_axis1(F32toU32_axis1,float4,read_imagef,\
                     convert_uint4, uint4, write_imageui)
REVERSESEQUENCE_axis1(F32toI32_axis1,float4,read_imagef,\
                     convert_int4,  int4,  write_imagei)
REVERSESEQUENCE_axis1(I32toF32_axis1,int4,  read_imagei,\
                     convert_float4,float4,write_imagef)
REVERSESEQUENCE_axis1(I32toU32_axis1,int4,  read_imagei,\
                     convert_uint4, uint4, write_imageui)
REVERSESEQUENCE_axis1(I32toI32_axis1,int4,  read_imagei,\
                     convert_int4,  int4,  write_imagei)
REVERSESEQUENCE_axis1(U32toF32_axis1,uint4, read_imageui,\
                     convert_float4,float4,write_imagef)
REVERSESEQUENCE_axis1(U32toU32_axis1,uint4, read_imageui,\
                     convert_uint4, uint4, write_imageui)
REVERSESEQUENCE_axis1(U32toI32_axis1,uint4, read_imageui,\
                      convert_int4,  int4,  write_imagei)

__kernel void reversesequence_BF16toBF16_axis1(
    __read_only  image2d_array_t  input0,
    __read_only  image2d_t        input1,
    __write_only image2d_array_t  output,
                 float            inoutScale,
                 float            inoutTail
    )
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);

    int4 coord_in = (int4)(gidx, gidy, gidz, 0);
    int4 coord_out = coord_in;
    uint4 src = read_imageui(input0, coord_in);
    int src_index = read_imagei(input1, (int2)(gidy, 0)).x;
    uint4 dst = src;
    if (gidz >= src_index)
    {
        write_imageui(output, coord_out, dst);
    }
    else
    {
        coord_out.z = src_index - 1 - coord_out.z;
        write_imageui(output, coord_out, dst);
    }
}
