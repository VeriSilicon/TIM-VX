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
    int stride_y;
#if (USE_40BITS_VA==0)
    int8 desc;
#else
    int8 desc;
    _viv_asm(GET_IMAGE_STRIDE, stride_y, input);
#endif
    _viv_asm(COPY, desc, input, sizeof(desc));
    uint address = as_uint(desc.s0);

#if (USE_40BITS_VA==0)
    stride_y = desc.s1;
#endif

    Image img =
    {
        .ptr                           = (uchar*)(uintptr_t)address,
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
    int8 desc;
    int2 strides;
    _viv_asm(COPY, desc, input, sizeof(desc));

#if (USE_40BITS_VA==0)
    strides.x = desc.s1;
    strides.y = desc.s4;
#else
    _viv_asm(GET_IMAGE_STRIDE, strides, input);
#endif
    uint address = as_uint(desc.s0);

    Tensor t =
    {
        .ptr                           = (uchar*)(uintptr_t)address,
        .stride_x                      = stride_x,
        .stride_y                      = strides.x,
        .stride_z                      = strides.y
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
