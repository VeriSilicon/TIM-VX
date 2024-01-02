#pragma OPENCL EXTENSION cl_viv_vx_extension : enable
#include "cl_viv_vx_ext.h"


_viv_uniform float width_scale;
_viv_uniform float height_scale;
_viv_uniform int   image_width;
_viv_uniform int   image_height;

#define CROP_AND_RESIZE_BILINEAR(name, read_type, dst_type, conv_type, write_type) \
__kernel void crop_and_resize_bilinear_##name \
( \
    __read_only image2d_array_t   input, \
    __read_only image2d_t         boxes, \
    __read_only image2d_t         box_ind, \
    __write_only image2d_array_t  output, \
                 uint             ori_depth, \
                 uint             ori_batchout, \
                 float            inOutScale, \
                 float            inOutTile, \
                 float            extrapolation_value \
) \
{ \
    int bb = get_global_id(2); \
    int y =  get_global_id(1); \
    int x = get_global_id(0); \
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int2 coord_box_ind = (int2)(bb, 0); \
    int b = read_imagei(box_ind, coord_box_ind).x; \
    float4 xy; \
    float in_x, in_y; \
    int d = 0; \
 \
    Image img_boxes = create_image_from_image2d(boxes, 2); \
    __global half* boxes_ptr = (__global half*)img_boxes.ptr; \
    xy = vload_half4(bb, boxes_ptr); \
    float _width_scale = convert_float(xy.w - xy.y) * width_scale; \
    float _height_scale = convert_float(xy.z - xy.x) * height_scale; \
    if (_width_scale == 0) xy.y = 0.5 * (xy.y + xy.w); \
    if (_height_scale == 0) xy.x = 0.5 * (xy.x + xy.z); \
    in_y = xy.x * convert_float(image_height - 1) + convert_float(y) * _height_scale; \
    in_x = xy.y * convert_float(image_width - 1) + convert_float(x) * _width_scale; \
    float y_lerp = in_y - floor(in_y); \
    float x_lerp = in_x - floor(in_x); \
    float4 src0, src1, src2, src3; \
    for (d = 0; d < ori_depth; d++) \
    { \
        int4 coord = (int4)(floor(in_x), floor(in_y), d + b * ori_depth, 0); \
        if (coord.x < 0 || coord.x > image_width - 1 || coord.y < 0 || coord.y > image_height - 1) \
        { \
            src0 = (float4)(extrapolation_value,0,0,0); \
        } \
        else \
        { \
            src0 = convert_float4(read_type(input, coord)); \
        } \
        coord.x = coord.x + 1; \
        if (coord.x < 0 || coord.x > image_width - 1 || coord.y < 0 || coord.y > image_height - 1) \
        { \
            src1 = (float4)(extrapolation_value,0,0,0); \
        } \
        else \
        { \
            src1 = convert_float4(read_type(input, coord)); \
        } \
        coord.y = coord.y + 1; \
        if (coord.x < 0 || coord.x > image_width - 1 || coord.y < 0 || coord.y > image_height - 1) \
        { \
            src3 = (float4)(extrapolation_value,0,0,0); \
        } \
        else \
        { \
            src3 = convert_float4(read_type(input, coord)); \
        } \
        coord.x = coord.x - 1; \
        if (coord.x < 0 || coord.x > image_width - 1 || coord.y < 0 || coord.y > image_height - 1) \
        { \
            src2 = (float4)(extrapolation_value,0,0,0); \
        } \
        else \
        { \
            src2 = convert_float4(read_type(input, coord)); \
        } \
        float4 top = src0 + (src1 - src0) * x_lerp; \
        float4 bottom = src2 + (src3 - src2) * x_lerp; \
        float4 value = top + (bottom - top) * y_lerp; \
        value = value * inOutScale + inOutTile; \
        dst_type dst = conv_type(value); \
        coord_out.z = d + coord_out.z * ori_depth; \
        write_type(output, coord_out, dst); \
    } \
}

CROP_AND_RESIZE_BILINEAR(U32toU32,read_imageui, \
uint4, convert_uint4, write_imageui)
CROP_AND_RESIZE_BILINEAR(U32toF32,read_imageui, \
float4,convert_float4,write_imagef)
CROP_AND_RESIZE_BILINEAR(F32toF32,read_imagef, \
float4, convert_float4,write_imagef)
CROP_AND_RESIZE_BILINEAR(F32toU32,read_imagef, \
uint4,  convert_uint4, write_imageui)
CROP_AND_RESIZE_BILINEAR(F32toI32,read_imagef, \
int4,   convert_int4,  write_imagei)
CROP_AND_RESIZE_BILINEAR(I32toI32,read_imagei,  \
int4,  convert_int4,  write_imagei)
CROP_AND_RESIZE_BILINEAR(I32toF32,read_imagei,  \
float4,convert_float4,write_imagef)