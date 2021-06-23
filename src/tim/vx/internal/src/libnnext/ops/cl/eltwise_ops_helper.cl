#pragma OPENCL EXTENSION CL_VIV_asm : enable
#pragma OPENCL EXTENSION cl_viv_vx_extension : enable

typedef struct Image
{
    __global uchar *ptr;
    int             stride_x;
    int             stride_y;
} Image;

inline uchar* get_image_ptr_from_coord(Image img, int2 coord)
{
    return img.ptr + coord.x * img.stride_x + coord.y * img.stride_y;
}

inline Image create_image_from_image2d(image2d_t input, int stride_x)
{
    int8 desc;
    _viv_asm(COPY, desc, input, sizeof(desc));

    Image img =
    {
        .ptr                           = (uchar*)desc.s0,
        .stride_x                      = stride_x,
        .stride_y                      = desc.s1
    };

    return img;
}

typedef struct Tensor
{
    __global uchar *ptr;
    int             stride_x;
    int             stride_y;
    int             stride_z;
} Tensor;

inline uchar* create_tensor_ptr_from_coord(Tensor t, int4 coord)
{
    return t.ptr + coord.x * t.stride_x + coord.y * t.stride_y + coord.z * t.stride_z;
}

inline Tensor create_tensor_from_image2d_array(image2d_array_t input, int stride_x)
{
    int8 desc;
    _viv_asm(COPY, desc, input, sizeof(desc));

    Tensor t =
    {
        .ptr                           = (uchar*)desc.s0,
        .stride_x                      = stride_x,
        .stride_y                      = desc.s1,
        .stride_z                      = desc.s4
    };

    return t;
}

#define readImage2DArray(Dest, Image, Coord)         \
    do {                                                       \
       int8 desc;                                              \
       _viv_asm(COPY, desc, Image, sizeof(desc));              \
       _viv_asm(CLAMP0MAX, (Coord).w, (Coord).z, desc.s5 - 1); \
       int baseAddr =  (int)(Coord).w * desc.s4 + desc.s0;     \
       _viv_asm(MOV, (Coord).w, baseAddr);                     \
       _viv_asm(IMAGE_READ_3D, Dest, Image, (Coord).xyww);     \
    } while (0)

#define writeImage2DArray(Image, Coord, Color)                 \
    do {                                                       \
       int8 desc;                                              \
       _viv_asm(COPY, desc, Image, sizeof(desc));              \
       _viv_asm(CLAMP0MAX, (Coord).w, (Coord).z, desc.s5 - 1); \
       int baseAddr =  (int)(Coord).w * desc.s4 + desc.s0;     \
       _viv_asm(MOV, (Coord).w, baseAddr);                     \
       _viv_asm(IMAGE_WRITE_3D, Color, Image, (Coord).xyww);   \
    } while (0)

#define readImage(Dest, Image, Coord)               \
    do {                                            \
       _viv_asm(IMAGE_READ, Dest, Image, Coord);    \
    } while (0)

#define writeImage(Image, Coord, Color)             \
    do {                                            \
       _viv_asm(IMAGE_WRITE, Color, Image, Coord);   \
    } while (0)
