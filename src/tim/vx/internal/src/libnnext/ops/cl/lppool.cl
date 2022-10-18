
#define LPPOOL_PROCESS(src_type, dst_type, readimage_type, conv_mode, writeimage_type) \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int hstart = gidy * stride_y - pad_top; \
    int wstart = gidx * stride_x - pad_left; \
    int hend = min(hstart + ksize_y, height); \
    int wend = min(wstart + ksize_x, width); \
    int4 coord_out = (int4)(gidx, gidy, get_global_id(2), 0); \
    int4 coord_in  = coord_out; \
    int h, w; \
    float sum_of_pow = 0; \
    dst_type out_data = (dst_type)(0); \
    src_type in_data; \
    float in_f32, out_f32; \
    hstart = max(hstart, 0); \
    wstart = max(wstart, 0); \
    for (h = hstart; h < hend; h++) \
    { \
        for (w = wstart; w < wend; w++) \
        { \
            coord_in.xy = (int2)(w, h); \
            in_data = readimage_type(input, coord_in).x; \
            in_f32 = convert_float(in_data) * inputScale + inputTail; \
            sum_of_pow += pow(fabs(in_f32),p); \
        } \
    } \
    out_f32 = pow(sum_of_pow, 1.0f / p) * outputScale + outputTail; \
    out_data.x = conv_mode(out_f32); \
    writeimage_type(output, coord_out, out_data); \

#define TENSOR_LPPOOL(src_name, dst_name, src_type, dst_type, readimage_type, conv_mode, writeimage_type) \
__kernel void lppool_##src_name##to##dst_name ( \
    __read_only image2d_array_t   input, \
    __write_only image2d_array_t  output, \
                 int              ksize_x, \
                 int              ksize_y, \
                 int              stride_x, \
                 int              stride_y, \
                 int              pad_left, \
                 int              pad_top, \
                 int              p, \
                 int              width, \
                 int              height, \
                 float            inputScale, \
                 float            inputTail, \
                 float            outputScale, \
                 float            outputTail) \
{ \
    LPPOOL_PROCESS(src_type, dst_type, readimage_type, conv_mode, writeimage_type); \
}

TENSOR_LPPOOL(F32, F32, float, float4, read_imagef, convert_float, write_imagef)
TENSOR_LPPOOL(F32, U32, float, uint4,  read_imagef, convert_uint,  write_imageui)
TENSOR_LPPOOL(F32, I32, float, int4,   read_imagef, convert_int,   write_imagei)

TENSOR_LPPOOL(U32, U32, uint, uint4,  read_imageui, convert_uint,  write_imageui)
TENSOR_LPPOOL(U32, F32, uint, float4, read_imageui, convert_float, write_imagef)
TENSOR_LPPOOL(U32, I32, uint, int4,   read_imageui, convert_int,   write_imagei)

TENSOR_LPPOOL(I32, I32, int, int4,    read_imagei, convert_int,   write_imagei)
TENSOR_LPPOOL(I32, F32, int, float4, read_imagei, convert_float, write_imagef)
TENSOR_LPPOOL(I32, U32, int, uint4,  read_imagei, convert_uint,  write_imageui)

__kernel void lppool_BF16toBF16(
    __read_only image2d_array_t   input,
    __write_only image2d_array_t  output,
                 int              ksize_x,
                 int              ksize_y,
                 int              stride_x,
                 int              stride_y,
                 int              pad_left,
                 int              pad_top,
                 int              p,
                 int              width,
                 int              height,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputTail)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int hstart = gidy * stride_y - pad_top;
    int wstart = gidx * stride_x - pad_left;
    int hend = min(hstart + ksize_y, height);
    int wend = min(wstart + ksize_x, width);
    int4 coord_out = (int4)(gidx, gidy, get_global_id(2), 0);
    int4 coord_in  = coord_out;
    int h, w;
    float sum_of_pow = 0;
    float out_data_f32 = 0;
    uint4 dst = (uint4)(0);
    float4 data_f32 = (float4)(0);
    uint4 data;
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    for (h = hstart; h < hend; h++)
    {
        for (w = wstart; w < wend; w++)
        {
            coord_in.xy = (int2)(w, h);
            data = read_imageui(input, coord_in);
            data = data << 16;
            _viv_asm(COPY, data_f32, data, 16);
            sum_of_pow += pow(abs(data_f32.x),p);
        }
    }
    out_data_f32 = pow(sum_of_pow, 1.0f / p);
    _viv_asm(COPY, dst, out_data_f32, 4);
    dst.x = dst.x >> 16;
    write_imageui(output, coord_out, dst);
}

