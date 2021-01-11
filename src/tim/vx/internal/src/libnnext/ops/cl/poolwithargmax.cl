
#define POOLWITHARGMAX_PROCESS(data_type, read_fun, write_fun0, write_fun1) \
    data_type src  = 0; \
    data_type max  = 0; \
    uint4  axis = 0; \
    src.x = read_fun(input, coord_in).x; \
    coord_in.x++; \
    src.y = read_fun(input, coord_in).x; \
    coord_in.y++; \
    src.w = read_fun(input, coord_in).x; \
    coord_in.x--; \
    src.z = read_fun(input, coord_in).x; \
    max.x  = src.x; \
    axis.x = 0; \
    if (src.y > max.x) \
    { \
        max.x  = src.y; \
        axis.x = 1; \
    } \
    if (src.z > max.x) \
    { \
        max.x  = src.z; \
        axis.x = 2; \
    } \
    if (src.w > max.x) \
    { \
        max.x  = src.w; \
        axis.x = 3; \
    } \
    write_fun0(output,  coord_out, max); \
    write_fun1(outaxis, coord_out, axis);


__kernel void poolwithargmax_F32to_F32_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   output,
    __write_only image2d_array_t   outaxis)
{
    int4 coord_out =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    POOLWITHARGMAX_PROCESS(float4, read_imagef, write_imagef, write_imageui)
}

__kernel void poolwithargmax_F32to_F32_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   output,
    __write_only image2d_t   outaxis)
{
    int2 coord_out =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in  =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    POOLWITHARGMAX_PROCESS(float4, read_imagef, write_imagef, write_imageui)
}

__kernel void poolwithargmax_I32to_I32_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   output,
    __write_only image2d_array_t   outaxis)
{
    int4 coord_out =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    POOLWITHARGMAX_PROCESS(int4, read_imagei, write_imagei, write_imageui)
}

__kernel void poolwithargmax_I32to_I32_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   output,
    __write_only image2d_t   outaxis)
{
    int2 coord_out =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in  =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    POOLWITHARGMAX_PROCESS(int4, read_imagei, write_imagei, write_imageui)
}


#define POOLWITHARGMAX_U8_PROCESS() \
    uint4 src  = 0; \
    uint4 max  = 0; \
    uint4 axis = 0; \
    float4 result = 0.0f; \
    src.x = read_imageui(input, coord_in).x; \
    coord_in.x++; \
    src.y = read_imageui(input, coord_in).x; \
    coord_in.y++; \
    src.w = read_imageui(input, coord_in).x; \
    coord_in.x--; \
    src.z = read_imageui(input, coord_in).x; \
    max.x  = src.x; \
    axis.x = 0; \
    if (src.y > max.x) \
    { \
        max.x  = src.y; \
        axis.x = 1; \
    } \
    if (src.z > max.x) \
    { \
        max.x  = src.z; \
        axis.x = 2; \
    } \
    if (src.w > max.x) \
    { \
        max.x  = src.w; \
        axis.x = 3; \
    } \
    result.x = convert_float4(max).x * scale_value + tail_value; \
    max = convert_uint4(result);\
    write_imageui(output,  coord_out, max); \
    write_imageui(outaxis, coord_out, axis);


__kernel void poolwithargmax_U8to_U8_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   output,
    __write_only image2d_array_t   outaxis,
                           float   scale_value,
                           float   tail_value)
{
    int4 coord_out =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    POOLWITHARGMAX_U8_PROCESS()
}

__kernel void poolwithargmax_U8to_U8_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   output,
    __write_only image2d_t   outaxis,
                     float   scale_value,
                     float   tail_value)
{
    int2 coord_out =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in  =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    POOLWITHARGMAX_U8_PROCESS()
}


#define POOLWITHARGMAX_U8_TO_F32_PROCESS() \
    uint4 src  = 0; \
    uint4 max  = 0; \
    uint4 axis = 0; \
    float4 result = 0.0f; \
    src.x = read_imageui(input, coord_in).x; \
    coord_in.x++; \
    src.y = read_imageui(input, coord_in).x; \
    coord_in.y++; \
    src.w = read_imageui(input, coord_in).x; \
    coord_in.x--; \
    src.z = read_imageui(input, coord_in).x; \
    max.x  = src.x; \
    axis.x = 0; \
    if (src.y > max.x) \
    { \
        max.x  = src.y; \
        axis.x = 1; \
    } \
    if (src.z > max.x) \
    { \
        max.x  = src.z; \
        axis.x = 2; \
    } \
    if (src.w > max.x) \
    { \
        max.x  = src.w; \
        axis.x = 3; \
    } \
    result.x = convert_float4(max).x * scale_value + tail_value; \
    write_imagef(output,  coord_out, result); \
    write_imageui(outaxis, coord_out, axis);


__kernel void poolwithargmax_U8to_F32_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   output,
    __write_only image2d_array_t   outaxis,
                           float   scale_value,
                           float   tail_value)
{
    int4 coord_out =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    POOLWITHARGMAX_U8_TO_F32_PROCESS()
}

__kernel void poolwithargmax_U8to_F32_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   output,
    __write_only image2d_t   outaxis,
                     float   scale_value,
                     float   tail_value)
{
    int2 coord_out =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in  =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    POOLWITHARGMAX_U8_TO_F32_PROCESS()
}

#define POOLWITHARGMAX_F32_TO_U8_PROCESS() \
    float4 src  = 0; \
    float4 max  = 0; \
    uint4 axis = 0; \
    uint4 dst  = 0; \
    float4 result = 0.0f; \
    src.x = read_imagef(input, coord_in).x; \
    coord_in.x++; \
    src.y = read_imagef(input, coord_in).x; \
    coord_in.y++; \
    src.w = read_imagef(input, coord_in).x; \
    coord_in.x--; \
    src.z = read_imagef(input, coord_in).x; \
    max.x  = src.x; \
    axis.x = 0; \
    if (src.y > max.x) \
    { \
        max.x  = src.y; \
        axis.x = 1; \
    } \
    if (src.z > max.x) \
    { \
        max.x  = src.z; \
        axis.x = 2; \
    } \
    if (src.w > max.x) \
    { \
        max.x  = src.w; \
        axis.x = 3; \
    } \
    result.x = max.x * scale_value + tail_value; \
    dst = convert_uint4(result);\
    write_imageui(output,  coord_out, dst); \
    write_imageui(outaxis, coord_out, axis);


__kernel void poolwithargmax_F32to_U8_U8(
    __read_only  image2d_array_t   input,
    __write_only image2d_array_t   output,
    __write_only image2d_array_t   outaxis,
                           float   scale_value,
                           float   tail_value)
{
    int4 coord_out =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in  =  (int4)(get_global_id(0) << 1, get_global_id(1) << 1, get_global_id(2), 0);
    POOLWITHARGMAX_F32_TO_U8_PROCESS()
}

__kernel void poolwithargmax_F32to_U8_U8_2D(
    __read_only  image2d_t   input,
    __write_only image2d_t   output,
    __write_only image2d_t   outaxis,
                     float   scale_value,
                     float   tail_value)
{
    int2 coord_out =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in  =  (int2)(get_global_id(0) << 1, get_global_id(1) << 1);
    POOLWITHARGMAX_F32_TO_U8_PROCESS()
}
