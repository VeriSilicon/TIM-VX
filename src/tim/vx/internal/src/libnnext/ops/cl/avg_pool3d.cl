#pragma OPENCL EXTENSION cl_viv_vx_extension : enable
#include "cl_viv_vx_ext.h"

#define TENSOR_AVG_POOL3D(src_name, dst_name, src_type, dst_type,\
                         readimage_type, conv_mode, writeimage_type) \
__kernel void avg_pool3d_##src_name##to##dst_name ( \
    __read_only image2d_array_t   input, \
    __write_only image2d_array_t  output, \
                 int              ksize_x, \
                 int              ksize_y, \
                 int              ksize_z, \
                 int              stride_x, \
                 int              stride_y, \
                 int              stride_z, \
                 int              pad_left, \
                 int              pad_top, \
                 int              pad_front, \
                 int              width, \
                 int              height, \
                 int              depth_in, \
                 int              depth_out, \
                 float            inputScale, \
                 float            inputTail, \
                 float            outputScale, \
                 float            outputTail, \
                 int              count_include_pad) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int offsetz = get_global_id(2); \
    int offsetz2 = offsetz / depth_out * depth_in; \
    int d, d2, h, w, count; \
    float sum = 0; \
    dst_type out_data = (dst_type)(0); \
    src_type in_data; \
    float in_f32, out_f32; \
    int wstart = gidx * stride_x - pad_left; \
    int hstart = gidy * stride_y - pad_top; \
    int wend = min(wstart + ksize_x, width); \
    int hend = min(hstart + ksize_y, height); \
    int dstart, dend; \
    int4 coord_in, coord_out; \
    wstart = max(wstart, 0); \
    hstart = max(hstart, 0); \
    for (d2 = 0; d2 < depth_out; d2++) \
    { \
        dstart = d2 * stride_z - pad_front; \
        dend = min(dstart + ksize_z, depth_in); \
        dstart = max(dstart, 0); \
        coord_out = (int4)(gidx, gidy, offsetz + d2, 0); \
        sum = 0; \
        count = 0; \
        for (d = dstart; d < dend; d++) \
        { \
            for (h = hstart; h < hend; h++) \
            { \
                for (w = wstart; w < wend; w++) \
                { \
                    coord_in = (int4)(w, h, d + offsetz2, 0); \
                    in_data = readimage_type(input, coord_in).x; \
                    in_f32 = convert_float(in_data) * inputScale + inputTail; \
                    sum += in_f32; \
                    count++; \
                } \
            } \
        } \
        if (count_include_pad == 1) \
        { \
            count = ksize_x * ksize_y * ksize_z; \
        } \
        out_f32 = (sum / count) * outputScale + outputTail; \
        out_data.x = conv_mode(out_f32); \
        writeimage_type(output, coord_out, out_data); \
    } \
}

TENSOR_AVG_POOL3D(F32, F32, float, float4, read_imagef, convert_float, write_imagef)
TENSOR_AVG_POOL3D(F32, U32, float, uint4,  read_imagef, convert_uint,  write_imageui)
TENSOR_AVG_POOL3D(F32, I32, float, int4,   read_imagef, convert_int,   write_imagei)

TENSOR_AVG_POOL3D(U32, U32, uint, uint4,  read_imageui, convert_uint,  write_imageui)
TENSOR_AVG_POOL3D(U32, F32, uint, float4, read_imageui, convert_float, write_imagef)
TENSOR_AVG_POOL3D(U32, I32, uint, int4,   read_imageui, convert_int,   write_imagei)

TENSOR_AVG_POOL3D(I32, I32, int, int4,    read_imagei, convert_int,   write_imagei)
TENSOR_AVG_POOL3D(I32, F32, int, float4, read_imagei, convert_float, write_imagef)
TENSOR_AVG_POOL3D(I32, U32, int, uint4,  read_imagei, convert_uint,  write_imageui)

__kernel void avg_pool3d_BF16toBF16 (
    __read_only image2d_array_t   input,
    __write_only image2d_array_t  output,
                 int              ksize_x,
                 int              ksize_y,
                 int              ksize_z,
                 int              stride_x,
                 int              stride_y,
                 int              stride_z,
                 int              pad_left,
                 int              pad_top,
                 int              pad_front,
                 int              width,
                 int              height,
                 int              depth_in,
                 int              depth_out,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputTail,
                 int              count_include_pad)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int offsetz = get_global_id(2);
    int offsetz2 = offsetz / depth_out * depth_in;
    int d, d2, h, w, count;
    float sum = 0;
    uint4 out_data = (uint4)(0);
    uint4 in_data;
    float in_f32, out_f32;
    int wstart = gidx * stride_x - pad_left;
    int hstart = gidy * stride_y - pad_top;
    int wend = min(wstart + ksize_x, width);
    int hend = min(hstart + ksize_y, height);
    int dstart, dend;
    int4 coord_in, coord_out;
    wstart = max(wstart, 0);
    hstart = max(hstart, 0);
    for (d2 = 0; d2 < depth_out; d2++)
    {
        dstart = d2 * stride_z - pad_front;
        dend = min(dstart + ksize_z, depth_in);
        dstart = max(dstart, 0);
        coord_out = (int4)(gidx, gidy, offsetz + d2, 0);
        sum = 0;
        count = 0;
        for (d = dstart; d < dend; d++)
        {
            for (h = hstart; h < hend; h++)
            {
                for (w = wstart; w < wend; w++)
                {
                    coord_in = (int4)(w, h, d + offsetz2, 0);
                    in_data = read_imageui(input, coord_in).x;
                    in_data = in_data << 16;
                    _viv_asm(COPY, in_f32, in_data, 16);
                    sum += in_f32;
                    count++;
                }
            }
        }
        if (count_include_pad == 1)
        {
            count = ksize_x * ksize_y * ksize_z;
        }
        out_f32 = sum / count;
        _viv_asm(COPY, out_data, out_f32, 4);
        out_data.x = out_data.x >> 16;
        write_imageui(output, coord_out, out_data);
    }
}