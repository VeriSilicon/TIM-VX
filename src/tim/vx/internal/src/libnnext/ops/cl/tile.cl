
#define TILE_3D(name0, name1, src_type, dst_type, conv_type, read_image_func, write_image_func) \
__kernel void tile_##name0##to##name1 \
    ( \
    __read_only  image2d_array_t input, \
    __write_only image2d_array_t output, \
                             int batchIn, \
                             int depthIn, \
                             int depthOut, \
                             int multiples_0, \
                             int multiples_1, \
                             int multiples_2, \
                             int multiples_3, \
                             float inoutscale, \
                             float inouttail \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int4 coord_out; \
    int width = get_image_width(input); \
    int height = get_image_height(input); \
 \
    src_type src; \
    dst_type dst; \
 \
    read_image_func(src, input, coord); \
 \
    int batch_id = (short)coord.z / (short)depthIn; \
    coord.z = (short)coord.z % (short)depthIn; \
    coord_out = coord; \
 \
    for (int w = 0; w < multiples_3; w++) \
    { \
        int batch = batchIn * w + batch_id; \
 \
        for(int z = 0; z < multiples_2; z++) \
        { \
            coord_out.z = coord.z + z * depthIn + batch * depthOut; \
 \
            for (int y = 0; y < multiples_1; y++) \
            { \
                coord_out.y = coord.y + y * height; \
 \
                for (int x = 0; x < multiples_0; x++) \
                { \
                    coord_out.x = coord.x + x * width; \
                    dst = conv_type(convert_float4(src) * inoutscale + inouttail); \
                    write_image_func(output, coord_out.xyzw, dst); \
                } \
            } \
        } \
    } \
}
TILE_3D(I32, I32, int4,   int4,  convert_int4_rte,  READ_IMAGEI_2DARRAY,  write_imagei)
TILE_3D(U32, U32, uint4,  uint4, convert_uint4_rte, READ_IMAGEUI_2DARRAY, write_imageui)
TILE_3D(F32, F32, float4, float4,convert_float4_rte,READ_IMAGEF_2DARRAY,  write_imagef)
TILE_3D(F32, U32, float4, uint4, convert_uint4_rte, READ_IMAGEF_2DARRAY,  write_imageui)

#define TILE_2D(name0, name1, src_type, dst_type, conv_type, read_image_func, write_image_func) \
__kernel void tile_##name0##to##name1##_2D \
    ( \
    __read_only  image2d_t input, \
    __write_only image2d_t output, \
                       int batchIn, \
                       int depthIn, \
                       int depthOut, \
                       int multiples_0, \
                       int multiples_1, \
                       int multiples_2, \
                       int multiples_3, \
                       float inoutscale, \
                       float inouttail \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
    int width = get_image_width(input); \
    int height = get_image_height(input); \
    int output_width = get_image_width(output); \
    int output_height = get_image_height(output); \
 \
    src_type src = read_image_func(input, coord); \
    dst_type dst; \
 \
    do \
    { \
        do \
        { \
            dst = conv_type(convert_float4(src) * inoutscale + inouttail); \
            write_image_func(output, coord, dst); \
            coord.x += width; \
        } while (coord.x < output_width); \
        coord.x = get_global_id(0); \
        coord.y += height; \
    } while (coord.y < output_height); \
}
TILE_2D(I32, I32, int4,   int4,  convert_int4_rte,  read_imagei,  write_imagei)
TILE_2D(U32, U32, uint4,  uint4, convert_uint4_rte, read_imageui, write_imageui)
TILE_2D(F32, F32, float4, float4,convert_float4_rte,read_imagef,  write_imagef)
TILE_2D(F32, U32, float4, uint4, convert_uint4_rte, read_imagef,  write_imageui)



