
#define TILE_3D(name0, name1, data_type, write_image_func) \
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
                             int multiples_3 \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    int4 coord_out; \
    int width = get_image_width(input); \
    int height = get_image_height(input); \
 \
    data_type src; \
    readImage2DArray(src, input, coord); \
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
                    write_image_func(output, coord_out.xyzw, src); \
                } \
            } \
        } \
    } \
}
TILE_3D(I32, I32, int4,   write_imagei)
TILE_3D(U32, U32, uint4,  write_imageui)
TILE_3D(F32, F32, float4, write_imagef)

#define TILE_2D(name0, name1, data_type) \
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
                       int multiples_3 \
    ) \
{ \
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \
    int width = get_image_width(input); \
    int height = get_image_height(input); \
    int output_width = get_image_width(output); \
    int output_height = get_image_height(output); \
 \
    data_type src; \
    readImage(src, input, coord); \
 \
    do \
    { \
        do \
        { \
            writeImage(output, coord, src); \
            coord.x += width; \
        } while (coord.x < output_width); \
        coord.x = get_global_id(0); \
        coord.y += height; \
    } while (coord.y < output_height); \
}
TILE_2D(I32, I32, int4)
TILE_2D(U32, U32, uint4)
TILE_2D(F32, F32, float4)



