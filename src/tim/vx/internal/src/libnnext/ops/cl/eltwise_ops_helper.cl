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

#if (USE_40BITS_VA==0)
    uint address = as_uint(desc.s0);
    int stride_y = desc.s1;
#else
    ulong address = as_ulong(desc.s05);
    int stride_y = desc.s6;
#endif

    Image img =
    {
        .ptr                           = (uchar*)address,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y
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

inline uchar* get_tensor_ptr_from_coord(Tensor t, int4 coord)
{
    return t.ptr + coord.x * t.stride_x + coord.y * t.stride_y + coord.z * t.stride_z;
}

inline Tensor create_tensor_from_image2d_array(image2d_array_t input, int stride_x)
{
#if (USE_40BITS_VA==0)
    int8 desc;
    _viv_asm(COPY, desc, input, sizeof(desc));

    uint address = as_uint(desc.s0);
    int stride_y = desc.s1;
    int stride_z = desc.s4;
#else
    int16 desc;
    _viv_asm(COPY, desc, input, sizeof(desc));

    ulong address = as_ulong(desc.s05);
    int stride_y = desc.s6;
    int stride_z = desc.sa;
#endif

    Tensor t =
    {
        .ptr                           = (uchar*)address,
        .stride_x                      = stride_x,
        .stride_y                      = stride_y,
        .stride_z                      = stride_z
    };

    return t;
}

#define READ_IMAGEF_2DARRAY(dest, tensor, coord) \
    do { \
        int depth = get_image_array_size(tensor); \
        int4 coord_in = coord; \
        _viv_asm(CLAMP0MAX, coord_in.z, coord_in.z, depth - 1); \
        dest = read_imagef(tensor, coord_in); \
       } while(0)

#define READ_IMAGEI_2DARRAY(dest, tensor, coord) \
    do { \
        int depth = get_image_array_size(tensor); \
        int4 coord_in = coord; \
        _viv_asm(CLAMP0MAX, coord_in.z, coord_in.z, depth - 1); \
        dest = read_imagei(tensor, coord_in); \
       } while(0)

#define READ_IMAGEUI_2DARRAY(dest, tensor, coord) \
    do { \
        int depth = get_image_array_size(tensor); \
        int4 coord_in = coord; \
        _viv_asm(CLAMP0MAX, coord_in.z, coord_in.z, depth - 1); \
        dest = read_imageui(tensor, coord_in); \
       } while(0)
