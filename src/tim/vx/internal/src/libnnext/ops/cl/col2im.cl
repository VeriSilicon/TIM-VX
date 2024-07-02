#pragma OPENCL EXTENSION cl_viv_vx_extension : enable
#include "cl_viv_vx_ext.h"

_viv_uniform int width_pad;
_viv_uniform int height_pad;
_viv_uniform int depth_pad;
_viv_uniform int move_time_x;
_viv_uniform int move_time_y;
_viv_uniform int kernel_x_new;
_viv_uniform int kernel_y_new;
_viv_uniform int kernel_z_new;
_viv_uniform int depth;

#define COL2IM(name, read_type, dst_type ,convert_type, write_type) \
__kernel void col2im_##name \
( \
    __read_only image2d_array_t   input, \
    __write_only image2d_array_t  output, \
                 int              stride_w, \
                 int              stride_h, \
                 int              stride_d, \
                 int              dilation_w, \
                 int              dilation_h, \
                 int              dilation_d, \
                 int              pad_w_front, \
                 int              pad_w_end, \
                 int              pad_h_front, \
                 int              pad_h_end, \
                 int              pad_d_front, \
                 int              pad_d_end, \
                 int              kernel_x, \
                 int              kernel_y, \
                 int              kernel_z, \
                 float            inOutScale, \
                 float            inOutTile \
) \
{ \
    int x = get_global_id(0); \
    int y = get_global_id(1); \
    int z = get_global_id(2); \
    int4 coord_out = (int4)(x,y,z,0); \
    int b = z / depth; \
    z = z % depth; \
    int4 coord_in = (int4)(0,0,b,0); \
 \
    float sum = 0.0f; \
    x = x + pad_w_front; \
    y = y + pad_h_front; \
    z = z + pad_d_front; \
    int offset_x = x % stride_w; \
    int offset_y = y % stride_h; \
    int offset_z = z % stride_d; \
    int i,j,k; \
    for (k = offset_z; k < kernel_z_new; k += stride_d) \
    { \
        if ((z - k) < 0 || (z + (kernel_z_new - k)) > depth_pad || k % dilation_d != 0) \
        { \
            continue; \
        } \
        for (j = offset_y; j < kernel_y_new; j = j + stride_h) \
        { \
            if ((y - j) < 0 || (y + (kernel_y_new - j)) > height_pad || j % dilation_h != 0) \
            { \
                continue; \
            } \
            for (i = offset_x; i < kernel_x_new; i = i + stride_w) \
            { \
                if ((x - i) < 0 || (x + (kernel_x_new - i)) > width_pad || i % dilation_w != 0) \
                { \
                    continue; \
                } \
                coord_in.x = (x - i + stride_w - 1) / stride_w + \
                             (y - j + stride_h - 1) / stride_h * move_time_x + \
                             (z - k + stride_d - 1) / stride_d * move_time_y * move_time_x; \
                coord_in.y = i / dilation_w + j * kernel_x / dilation_h + k * kernel_x * kernel_y / dilation_d; \
                sum = sum + convert_float(read_type(input, coord_in).x); \
            } \
        } \
    } \
    sum = sum * inOutScale + inOutTile; \
    dst_type dst = 0; \
    dst.x = convert_type(sum); \
    write_type(output, coord_out, dst); \
}
COL2IM(U32toU32, read_imageui, uint4,  convert_uint,  write_imageui)
COL2IM(U32toI32, read_imageui, int4,   convert_int,   write_imagei)
COL2IM(U32toF32, read_imageui, float4, convert_float, write_imagef)
COL2IM(I32toU32, read_imagei,  uint4,  convert_uint,  write_imageui)
COL2IM(I32toI32, read_imagei,  int4,   convert_int,   write_imagei)
COL2IM(I32toF32, read_imagei,  float4, convert_float, write_imagef)
COL2IM(F32toU32, read_imagef,  uint4,  convert_uint,  write_imageui)
COL2IM(F32toI32, read_imagef,  int4,   convert_int,   write_imagei)
COL2IM(F32toF32, read_imagef,  float4, convert_float, write_imagef)

#define COL2IM_2D(name, read_type, dst_type ,convert_type, write_type) \
__kernel void col2im_##name##_2D \
( \
    __read_only image2d_array_t   input, \
    __write_only image2d_array_t  output, \
                 int              stride_w, \
                 int              stride_h, \
                 int              stride_d, \
                 int              dilation_w, \
                 int              dilation_h, \
                 int              dilation_d, \
                 int              pad_w_front, \
                 int              pad_w_end, \
                 int              pad_h_front, \
                 int              pad_h_end, \
                 int              pad_d_front, \
                 int              pad_d_end, \
                 int              kernel_x, \
                 int              kernel_y, \
                 int              kernel_z, \
                 float            inOutScale, \
                 float            inOutTile \
) \
{ \
    int x = get_global_id(0); \
    int y = get_global_id(1); \
    int z = get_global_id(2); \
    int4 coord_out = (int4)(x,y,z,0); \
    int4 coord_in = (int4)(0,0,z,0); \
 \
    float sum = 0.0f; \
    x = x + pad_w_front; \
    y = y + pad_h_front; \
    int offset_x = x % stride_w; \
    int offset_y = y % stride_h; \
    int i,j; \
    for (j = offset_y; j < kernel_y_new; j = j + stride_h) \
    { \
        if ((y - j) < 0 || (y + (kernel_y_new - j)) > height_pad || j % dilation_h != 0) \
        { \
            continue; \
        } \
        for (i = offset_x; i < kernel_x_new; i = i + stride_w) \
        { \
            if ((x - i) < 0 || (x + (kernel_x_new - i)) > width_pad || i % dilation_w != 0) \
            { \
                continue; \
            } \
            coord_in.x = (x - i + stride_w - 1) / stride_w + \
                         (y - j + stride_h - 1) / stride_h * move_time_x; \
            coord_in.y = i / dilation_w + j * kernel_x / dilation_h; \
            sum = sum + convert_float(read_type(input, coord_in).x); \
        } \
    } \
    sum = sum * inOutScale + inOutTile; \
    dst_type dst = 0; \
    dst.x = convert_type(sum); \
    write_type(output, coord_out, dst); \
}
COL2IM_2D(U32toU32, read_imageui, uint4,  convert_uint,  write_imageui)
COL2IM_2D(U32toI32, read_imageui, int4,   convert_int,   write_imagei)
COL2IM_2D(U32toF32, read_imageui, float4, convert_float, write_imagef)
COL2IM_2D(I32toU32, read_imagei,  uint4,  convert_uint,  write_imageui)
COL2IM_2D(I32toI32, read_imagei,  int4,   convert_int,   write_imagei)
COL2IM_2D(I32toF32, read_imagei,  float4, convert_float, write_imagef)
COL2IM_2D(F32toU32, read_imagef,  uint4,  convert_uint,  write_imageui)
COL2IM_2D(F32toI32, read_imagef,  int4,   convert_int,   write_imagei)
COL2IM_2D(F32toF32, read_imagef,  float4, convert_float, write_imagef)