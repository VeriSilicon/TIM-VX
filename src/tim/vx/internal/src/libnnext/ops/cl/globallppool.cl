
#define GLOBALLPPOOL_PROCESS(src_type, dst_type, readimage_type, conv_mode, writeimage_type) \
    int gidx = get_global_id(0); \
    int4 coord_out = (int4)(0, 0, gidx, 0); \
    int4 coord_in  = coord_out; \
    int h, w; \
    float sum_of_pow = 0; \
    dst_type out_data = (dst_type)(0); \
    src_type in_data; \
    float in_f32, out_f32; \
    for (h = 0; h < height; h++) \
    { \
        for (w = 0; w < width; w++) \
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

#define TENSOR_GLOBALLPPOOL(src_name, dst_name, src_type, dst_type, readimage_type, conv_mode, writeimage_type) \
__kernel void globallppool_##src_name##to##dst_name ( \
    __read_only image2d_array_t   input, \
    __write_only image2d_array_t  output, \
                 int              p, \
                 int              width, \
                 int              height, \
                 float            inputScale, \
                 float            inputTail, \
                 float            outputScale, \
                 float            outputTail) \
{ \
    GLOBALLPPOOL_PROCESS(src_type, dst_type, readimage_type, conv_mode, writeimage_type); \
}

TENSOR_GLOBALLPPOOL(F32, F32, float, float4, read_imagef, convert_float, write_imagef)
TENSOR_GLOBALLPPOOL(F32, U32, float, uint4,  read_imagef, convert_uint,  write_imageui)
TENSOR_GLOBALLPPOOL(F32, I32, float, int4,   read_imagef, convert_int,   write_imagei)

TENSOR_GLOBALLPPOOL(U32, U32, uint, uint4,  read_imageui, convert_uint,  write_imageui)
TENSOR_GLOBALLPPOOL(U32, F32, uint, float4, read_imageui, convert_float, write_imagef)
TENSOR_GLOBALLPPOOL(U32, I32, uint, int4,   read_imageui, convert_int,   write_imagei)

TENSOR_GLOBALLPPOOL(I32, I32, int, int4,    read_imagei, convert_int,   write_imagei)
TENSOR_GLOBALLPPOOL(I32, F32, int, float4, read_imagei, convert_float, write_imagef)
TENSOR_GLOBALLPPOOL(I32, U32, int, uint4,  read_imagei, convert_uint,  write_imageui)

__kernel void globallppool_BF16toBF16(
    __read_only image2d_array_t   input,
    __write_only image2d_array_t  output,
                 int              p,
                 int              width,
                 int              height,
                 float            inputScale,
                 float            inputTail,
                 float            outputScale,
                 float            outputTail)
{
    int gidx = get_global_id(0);
    int4 coord_out = (int4)(1, 1, gidx , 0);
    int4 coord_in  = coord_out;
    int h, w;
    float sum_of_pow = 0;
    float out_data_f32 = 0;
    uint4 dst = (uint4)(0);
    float4 data_f32 = (float4)(0);
    uint4 data;

    for (h = 0; h < height; h++)
    {
        for (w = 0; w < width; w++)
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

