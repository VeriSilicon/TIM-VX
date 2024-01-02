#pragma OPENCL EXTENSION cl_viv_vx_extension : enable
#include "cl_viv_vx_ext.h"


_viv_uniform float width_scale;
_viv_uniform float height_scale;
_viv_uniform int   image_width;
_viv_uniform int   image_height;

#define CROP_AND_RESIZE_NEAREST_NEIGHTBOR(name,src_type, read_type, dst_type, conv_type, write_type) \
__kernel void crop_and_resize_nearest_neighbor_##name \
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
    int in_x, in_y, d = 0; \
 \
    Image img_boxes = create_image_from_image2d(boxes, 2); \
    __global half* boxes_ptr = (__global half*)img_boxes.ptr; \
    xy = vload_half4(bb, boxes_ptr); \
    float _width_scale = convert_float(xy.w - xy.y) * width_scale; \
    float _height_scale = convert_float(xy.z - xy.x) * height_scale; \
    if (_width_scale == 0) xy.y = 0.5 * (xy.y + xy.w); \
    if (_height_scale == 0) xy.x = 0.5 * (xy.x + xy.z); \
    in_y = convert_int(round(xy.x * convert_float(image_height - 1) \
                                  + convert_float(y) * _height_scale)); \
    in_x = convert_int(round(xy.y * convert_float(image_width - 1) \
                                  + convert_float(x) * _width_scale)); \
    for (d = 0; d < ori_depth; d++) \
    { \
        int4 coord = (int4)(in_x, in_y, d + b * ori_depth, 0); \
        float4 src_f; \
        if (coord.x < 0 || coord.x > image_width - 1 || coord.y < 0 || coord.y > image_height - 1) \
        { \
            src_f = (float4)(extrapolation_value, 0, 0, 0); \
        } \
        else \
        { \
            src_type src = read_type(input, coord); \
            src_f = convert_float4(src); \
        } \
        src_f = src_f * inOutScale + inOutTile; \
        dst_type dst = conv_type(src_f); \
        coord_out.z = d + coord_out.z * ori_depth; \
        write_type(output, coord_out, dst); \
    } \
}

CROP_AND_RESIZE_NEAREST_NEIGHTBOR(U32toU32,uint4, \
read_imageui, uint4, convert_uint4, write_imageui)
CROP_AND_RESIZE_NEAREST_NEIGHTBOR(U32toF32,uint4, \
read_imageui, float4,convert_float4,write_imagef)
CROP_AND_RESIZE_NEAREST_NEIGHTBOR(F32toF32,float4, \
read_imagef, float4,convert_float4,write_imagef)
CROP_AND_RESIZE_NEAREST_NEIGHTBOR(F32toU32,float4, \
read_imagef, uint4, convert_uint4, write_imageui)
CROP_AND_RESIZE_NEAREST_NEIGHTBOR(F32toI32,float4, \
read_imagef, int4,  convert_int4,  write_imagei)
CROP_AND_RESIZE_NEAREST_NEIGHTBOR(I32toI32,int4,  \
read_imagei,  int4,  convert_int4,  write_imagei)
CROP_AND_RESIZE_NEAREST_NEIGHTBOR(I32toF32,int4,  \
read_imagei,  float4,convert_float4,write_imagef)