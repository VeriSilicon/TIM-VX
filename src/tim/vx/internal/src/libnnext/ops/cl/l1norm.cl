#define eps 1e-12

#define TENSOR_L1NORM_axis0(src_name, dst_name, src_type, dst_type, \
                      readimage_type, conv_mode, writeimage_type) \
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void l1norm_##src_name##to##dst_name##_axis0( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output, \
                 float            inputZp, \
                 float            outputscale, \
                 float            outputtail, \
                 int              axis, \
                 int              axis_size) \
{ \
    int lidx = get_local_id(0); \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int gidz = get_global_id(2); \
    src_type src; \
    dst_type dst; \
    float4 src_f, dst_f; \
    float sum = 0; \
    float rcp_sum = 0; \
    int4 coord= (int4)(gidx, gidy, gidz, 0); \
    __local float lcl_sum[16]; \
    for (; coord.x < axis_size; coord.x += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        sum += fabs(src_f.x); \
    } \
    lcl_sum[lidx] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float4 *pLocalPtr = (float4 *)&lcl_sum[0]; \
    float4 one = (float4)(1, 1, 1, 1); \
    float4 data0; \
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3]; \
    rcp_sum =  1 / (dot(data0, one) + eps); \
    for (coord.x = gidx; coord.x < axis_size; coord.x += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        dst_f = src_f * rcp_sum; \
        dst = conv_mode(dst_f * outputscale + outputtail); \
        writeimage_type(output, coord, dst); \
    } \
}

TENSOR_L1NORM_axis0(F32,F32,float4,float4,read_imagef, convert_float4,write_imagef)
TENSOR_L1NORM_axis0(U32,U32,uint4, uint4, read_imageui,convert_uint4, write_imageui)
TENSOR_L1NORM_axis0(I32,I32,int4,  int4,  read_imagei, convert_int4,  write_imagei)
TENSOR_L1NORM_axis0(F32,U32,float4,uint4, read_imagef, convert_uint4, write_imageui)
TENSOR_L1NORM_axis0(F32,I32,float4,int4,  read_imagef, convert_int4,  write_imagei)
TENSOR_L1NORM_axis0(U32,F32,uint4, float4,read_imageui,convert_float4,write_imagef)
TENSOR_L1NORM_axis0(I32,F32,int4,  float4,read_imagei, convert_float4,write_imagef)

#define TENSOR_L1NORM_axis1(src_name, dst_name, src_type, dst_type, \
                      readimage_type, conv_mode, writeimage_type) \
__kernel __attribute__((reqd_work_group_size(1, 16, 1))) void l1norm_##src_name##to##dst_name##_axis1( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output, \
                 float            inputZp, \
                 float            outputscale, \
                 float            outputtail, \
                 int              axis, \
                 int              axis_size) \
{ \
    int lidy = get_local_id(1); \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int gidz = get_global_id(2); \
    src_type src; \
    dst_type dst; \
    float4 src_f, dst_f; \
    float sum = 0; \
    float rcp_sum = 0; \
    int4 coord= (int4)(gidx, gidy, gidz, 0); \
    __local float lcl_sum[16]; \
    for (; coord.y < axis_size; coord.y += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        sum += fabs(src_f.x); \
    } \
    lcl_sum[lidy] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float4 *pLocalPtr = (float4 *)&lcl_sum[0]; \
    float4 one = (float4)(1, 1, 1, 1); \
    float4 data0; \
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3]; \
    rcp_sum =  1 / (dot(data0, one) + eps); \
    for (coord.y = gidy; coord.y < axis_size; coord.y += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        dst_f = src_f * rcp_sum; \
        dst = conv_mode(dst_f * outputscale + outputtail); \
        writeimage_type(output, coord, dst); \
    } \
}

TENSOR_L1NORM_axis1(F32,F32,float4,float4,read_imagef, convert_float4,write_imagef)
TENSOR_L1NORM_axis1(U32,U32,uint4, uint4, read_imageui,convert_uint4, write_imageui)
TENSOR_L1NORM_axis1(I32,I32,int4,  int4,  read_imagei, convert_int4,  write_imagei)
TENSOR_L1NORM_axis1(F32,U32,float4,uint4, read_imagef, convert_uint4, write_imageui)
TENSOR_L1NORM_axis1(F32,I32,float4,int4,  read_imagef, convert_int4,  write_imagei)
TENSOR_L1NORM_axis1(U32,F32,uint4, float4,read_imageui,convert_float4,write_imagef)
TENSOR_L1NORM_axis1(I32,F32,int4,  float4,read_imagei, convert_float4,write_imagef)

#define TENSOR_L1NORM_axis2(src_name, dst_name, src_type, dst_type, \
                      readimage_type, conv_mode, writeimage_type) \
__kernel __attribute__((reqd_work_group_size(1, 1, 16))) void l1norm_##src_name##to##dst_name##_axis2( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output, \
                 float            inputZp, \
                 float            outputscale, \
                 float            outputtail, \
                 int              axis, \
                 int              axis_size) \
{ \
    int lidz = get_local_id(2); \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int gidz = get_global_id(2); \
    src_type src; \
    dst_type dst; \
    float4 src_f, dst_f; \
    float sum = 0; \
    float rcp_sum = 0; \
    int4 coord= (int4)(gidx, gidy, gidz, 0); \
    __local float lcl_sum[16]; \
    for (; coord.z < axis_size; coord.z += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        sum += fabs(src_f.x); \
    } \
    lcl_sum[lidz] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float4 *pLocalPtr = (float4 *)&lcl_sum[0]; \
    float4 one = (float4)(1, 1, 1, 1); \
    float4 data0; \
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3]; \
    rcp_sum =  1 / (dot(data0, one) + eps); \
    for (coord.z = gidz; coord.z < axis_size; coord.z += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        dst_f = src_f * rcp_sum; \
        dst = conv_mode(dst_f * outputscale + outputtail); \
        writeimage_type(output, coord, dst); \
    } \
}

TENSOR_L1NORM_axis2(F32,F32,float4,float4,read_imagef, convert_float4,write_imagef)
TENSOR_L1NORM_axis2(U32,U32,uint4, uint4, read_imageui,convert_uint4, write_imageui)
TENSOR_L1NORM_axis2(I32,I32,int4,  int4,  read_imagei, convert_int4,  write_imagei)
TENSOR_L1NORM_axis2(F32,U32,float4,uint4, read_imagef, convert_uint4, write_imageui)
TENSOR_L1NORM_axis2(F32,I32,float4,int4,  read_imagef, convert_int4,  write_imagei)
TENSOR_L1NORM_axis2(U32,F32,uint4, float4,read_imageui,convert_float4,write_imagef)
TENSOR_L1NORM_axis2(I32,F32,int4,  float4,read_imagei, convert_float4,write_imagef)

#define TENSOR_L1NORM_2D_axis0(src_name, dst_name, src_type, dst_type,\
          readimage_type, conv_mode, writeimage_type) \
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void l1norm_##src_name##to##dst_name##_2D_axis0( \
    __read_only  image2d_t        input, \
    __write_only image2d_t        output, \
                 float            inputZp, \
                 float            outputscale, \
                 float            outputtail, \
                 int              axis, \
                 int              axis_size) \
{ \
    int lidx = get_local_id(0); \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    src_type src; \
    dst_type dst; \
    float4 src_f, dst_f; \
    float sum = 0; \
    float rcp_sum = 0; \
    int2 coord = (int2)(gidx, gidy); \
    __local float lcl_sum[16]; \
    for (; coord.x < axis_size; coord.x += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        sum += fabs(src_f.x); \
    } \
    lcl_sum[lidx] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float4 *pLocalPtr = (float4 *)&lcl_sum[0]; \
    float4 one = (float4)(1, 1, 1, 1); \
    float4 data0; \
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3]; \
    rcp_sum = 1 / (dot(data0, one) + eps); \
    for (coord.x = gidx; coord.x < axis_size; coord.x += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        dst_f = src_f * rcp_sum; \
        dst = conv_mode(dst_f * outputscale + outputtail); \
        writeimage_type(output, coord, dst); \
    } \
}

TENSOR_L1NORM_2D_axis0(F32,F32,float4,float4,read_imagef, convert_float4,write_imagef)
TENSOR_L1NORM_2D_axis0(U32,U32,uint4, uint4, read_imageui,convert_uint4, write_imageui)
TENSOR_L1NORM_2D_axis0(I32,I32,int4,  int4,  read_imagei, convert_int4,  write_imagei)
TENSOR_L1NORM_2D_axis0(F32,U32,float4,uint4, read_imagef, convert_uint4, write_imageui)
TENSOR_L1NORM_2D_axis0(F32,I32,float4,int4,  read_imagef, convert_int4,  write_imagei)
TENSOR_L1NORM_2D_axis0(U32,F32,uint4, float4,read_imageui,convert_float4,write_imagef)
TENSOR_L1NORM_2D_axis0(I32,F32,int4,  float4,read_imagei, convert_float4,write_imagef)


#define TENSOR_L1NORM_2D_axis1(src_name, dst_name, src_type, dst_type,\
             readimage_type, conv_mode, writeimage_type) \
__kernel __attribute__((reqd_work_group_size(1, 16, 1))) void l1norm_##src_name##to##dst_name##_2D_axis1( \
    __read_only  image2d_t        input, \
    __write_only image2d_t        output, \
                 float            inputZp, \
                 float            outputscale, \
                 float            outputtail, \
                 int              axis, \
                 int              axis_size) \
{ \
    int lidy = get_local_id(1); \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    src_type src; \
    dst_type dst; \
    float4 src_f, dst_f; \
    float sum = 0; \
    float rcp_sum = 0; \
    int2 coord = (int2)(gidx, gidy); \
    __local float lcl_sum[16]; \
    for (; coord.y < axis_size; coord.y += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        sum += fabs(src_f.x); \
    } \
    lcl_sum[lidy] = sum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float4 *pLocalPtr = (float4 *)&lcl_sum[0]; \
    float4 one = (float4)(1, 1, 1, 1); \
    float4 data0; \
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3]; \
    rcp_sum = 1 / (dot(data0, one) + eps); \
    for (coord.y = gidy; coord.y < axis_size; coord.y += 16) \
    { \
        src = readimage_type(input, coord); \
        src_f = convert_float4(src) - inputZp; \
        dst_f = src_f * rcp_sum; \
        dst = conv_mode(dst_f * outputscale + outputtail); \
        writeimage_type(output, coord, dst); \
    } \
}

TENSOR_L1NORM_2D_axis1(F32,F32,float4,float4,read_imagef, convert_float4,write_imagef)
TENSOR_L1NORM_2D_axis1(U32,U32,uint4, uint4, read_imageui,convert_uint4, write_imageui)
TENSOR_L1NORM_2D_axis1(I32,I32,int4,  int4,  read_imagei, convert_int4,  write_imagei)
TENSOR_L1NORM_2D_axis1(F32,U32,float4,uint4, read_imagef, convert_uint4, write_imageui)
TENSOR_L1NORM_2D_axis1(F32,I32,float4,int4,  read_imagef, convert_int4,  write_imagei)
TENSOR_L1NORM_2D_axis1(U32,F32,uint4, float4,read_imageui,convert_float4,write_imagef)
TENSOR_L1NORM_2D_axis1(I32,F32,int4,  float4,read_imagei, convert_float4,write_imagef)