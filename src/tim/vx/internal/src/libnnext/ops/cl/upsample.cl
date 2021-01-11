
#define UPSAMPLE_PROCESS(data_type, read_fun, write_fun) \
    data_type src  = 0; \
    data_type dst  = 0; \
    uint4  axis = 0; \
    src.x  = read_fun(input,  coord_in).x; \
    axis.x = read_imageui(inaxis, coord_in).x; \
    dst.x = axis.x == 0 ? src.x : 0; \
    write_fun(output,  coord_out, dst); \
    dst.x = axis.x == 1 ? src.x : 0; \
    coord_out.x++; \
    write_fun(output,  coord_out, dst); \
    dst.x = axis.x == 3 ? src.x : 0; \
    coord_out.y++; \
    write_fun(output,  coord_out, dst); \
    dst.x = axis.x == 2 ? src.x : 0; \
    coord_out.x--; \
    write_fun(output,  coord_out, dst);


__kernel void upsample_F32_U8to_F32(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   inaxis,
    __write_only image2d_array_t   output)
{
    int4 coord_out =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    UPSAMPLE_PROCESS(float4, read_imagef, write_imagef)
}

__kernel void upsample_F32_U8to_F32_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   inaxis,
    __write_only image2d_t   output)
{
    int2 coord_out =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    int2 coord_in  =  (int2)(get_global_id(0), get_global_id(1));
    UPSAMPLE_PROCESS(float4, read_imagef, write_imagef)
}

__kernel void upsample_I32_U8to_I32(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   inaxis,
    __write_only image2d_array_t   output)
{
    int4 coord_out =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    UPSAMPLE_PROCESS(int4, read_imagei, write_imagei)
}

__kernel void upsample_I32_U8to_I32_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   inaxis,
    __write_only image2d_t   output)
{
    int2 coord_out =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    int2 coord_in  =  (int2)(get_global_id(0), get_global_id(1));
    UPSAMPLE_PROCESS(int4, read_imagei, write_imagei)
}


#define UPSAMPLE_U8_PROCESS() \
    uint4  src  = 0; \
    uint4  dst  = 0; \
    uint4  axis = 0; \
    float4 result = 0.0f; \
    uint   output_zp = (uint)zp_out; \
    src.x  = read_imageui(input,  coord_in).x; \
    axis.x = read_imageui(inaxis, coord_in).x; \
    result.x = convert_float4(src).x * scale_value + tail_value; \
    src = convert_uint4(result);\
    dst.x = axis.x == 0 ? src.x : output_zp; \
    write_imageui(output,  coord_out, dst); \
    dst.x = axis.x == 1 ? src.x : output_zp; \
    coord_out.x++; \
    write_imageui(output,  coord_out, dst); \
    dst.x = axis.x == 3 ? src.x : output_zp; \
    coord_out.y++; \
    write_imageui(output,  coord_out, dst); \
    dst.x = axis.x == 2 ? src.x : output_zp; \
    coord_out.x--; \
    write_imageui(output,  coord_out, dst);


__kernel void upsample_U8_U8to_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   inaxis,
    __write_only image2d_array_t   output,
                           float   scale_value,
                           float   tail_value,
                             int   zp_out)
{
    int4 coord_out =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    UPSAMPLE_U8_PROCESS()
}

__kernel void upsample_U8_U8to_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   inaxis,
    __write_only image2d_t   output,
                     float   scale_value,
                     float   tail_value,
                       int   zp_out)
{
    int2 coord_out =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    int2 coord_in  =  (int2)(get_global_id(0), get_global_id(1));
    UPSAMPLE_U8_PROCESS()
}

#define UPSAMPLE_U8_TO_F32PROCESS() \
    uint4  src  = 0; \
    float4  dst  = 0; \
    uint4  axis = 0; \
    float4 result = 0.0f; \
    src.x  = read_imageui(input,  coord_in).x; \
    axis.x = read_imageui(inaxis, coord_in).x; \
    result.x = convert_float4(src).x * scale_value + tail_value; \
    dst.x = axis.x == 0 ? result.x : 0.0f; \
    write_imagef(output,  coord_out, dst); \
    dst.x = axis.x == 1 ? result.x : 0.0f; \
    coord_out.x++; \
    write_imagef(output,  coord_out, dst); \
    dst.x = axis.x == 3 ? result.x : 0.0f; \
    coord_out.y++; \
    write_imagef(output,  coord_out, dst); \
    dst.x = axis.x == 2 ? result.x : 0.0f; \
    coord_out.x--; \
    write_imagef(output,  coord_out, dst);


__kernel void upsample_U8_U8to_F32(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   inaxis,
    __write_only image2d_array_t   output,
                           float   scale_value,
                           float   tail_value,
                             int   zp_out)
{
    int4 coord_out =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    UPSAMPLE_U8_TO_F32PROCESS()
}

__kernel void upsample_U8_U8to_F32_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   inaxis,
    __write_only image2d_t   output,
                     float   scale_value,
                     float   tail_value,
                       int   zp_out)
{
    int2 coord_out =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    int2 coord_in  =  (int2)(get_global_id(0), get_global_id(1));
    UPSAMPLE_U8_TO_F32PROCESS()
}


#define UPSAMPLE_F32_TO_U8_PROCESS() \
    uint4  src  = 0; \
    uint4  dst  = 0; \
    uint4  axis = 0; \
    float4 result = 0.0f; \
    uint   output_zp = (uint)zp_out; \
    result.x  = read_imagef(input,  coord_in).x; \
    axis.x = read_imageui(inaxis, coord_in).x; \
    result.x = result.x * scale_value + tail_value; \
    src = convert_uint4(result);\
    dst.x = axis.x == 0 ? src.x : output_zp; \
    write_imageui(output,  coord_out, dst); \
    dst.x = axis.x == 1 ? src.x : output_zp; \
    coord_out.x++; \
    write_imageui(output,  coord_out, dst); \
    dst.x = axis.x == 3 ? src.x : output_zp; \
    coord_out.y++; \
    write_imageui(output,  coord_out, dst); \
    dst.x = axis.x == 2 ? src.x : output_zp; \
    coord_out.x--; \
    write_imageui(output,  coord_out, dst);


__kernel void upsample_F32_U8to_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   inaxis,
    __write_only image2d_array_t   output,
                           float   scale_value,
                           float   tail_value,
                             int   zp_out)
{
    int4 coord_out =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    UPSAMPLE_F32_TO_U8_PROCESS()
}

__kernel void upsample_F32_U8to_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   inaxis,
    __write_only image2d_t   output,
                     float   scale_value,
                     float   tail_value,
                       int   zp_out)
{
    int2 coord_out =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    int2 coord_in  =  (int2)(get_global_id(0), get_global_id(1));
    UPSAMPLE_F32_TO_U8_PROCESS()
}
