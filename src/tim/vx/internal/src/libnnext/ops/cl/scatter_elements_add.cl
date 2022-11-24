
#define SE_ADD_AXIS0_32BITS_IMPL(name, dtype) \
__kernel void scatter_elements_add_axis0_##name \
    ( \
    __read_only  image2d_t ref, \
    __read_only  image2d_t indices, \
    __read_only  image2d_t update, \
    __write_only image2d_t output, \
                 int       axis, \
                 int       reduction, \
                 float     ref_scale, \
                 float     ref_tail, \
                 float     update_scale, \
                 float     update_tail, \
                 float     output_zp, \
                 int       inner_size, \
                 int       axis_size, \
                 int       outer_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \
 \
    Image ref_i = create_image_from_image2d(ref, 4); \
    Image update_i = create_image_from_image2d(update, 4); \
    Image indices_i = create_image_from_image2d(indices, 4); \
    Image output_i = create_image_from_image2d(output, 4); \
 \
    dtype *ref_ptr = (dtype *)get_image_ptr_from_coord(ref_i, coord.xy); \
    dtype *output_ptr = (dtype *)get_image_ptr_from_coord(output_i, coord.xy); \
    dtype data = ref_ptr[0]; \
    if (coord.y < outer_size) \
    { \
        dtype *update_ptr = (dtype *)get_image_ptr_from_coord(update_i, coord.wy); \
        int *indices_ptr = (int *)get_image_ptr_from_coord(indices_i, coord.wy); \
        for(int x = 0; x < axis_size; x ++) \
        { \
            int offset = indices_ptr[x]; \
            if (offset == coord.x) \
            { \
                data += update_ptr[x]; \
            } \
        } \
    } \
 \
    output_ptr[0] = data; \
}
SE_ADD_AXIS0_32BITS_IMPL(F32_I32_F32toF32, float)
SE_ADD_AXIS0_32BITS_IMPL(I32_I32_I32toI32, int)

#define SE_ADD_AXIS0_16BITS_IMPL(name, dtype, conv_func) \
__kernel void scatter_elements_add_axis0_##name \
    ( \
    __read_only  image2d_t ref, \
    __read_only  image2d_t indices, \
    __read_only  image2d_t update, \
    __write_only image2d_t output, \
                 int       axis, \
                 int       reduction, \
                 float     ref_scale, \
                 float     ref_tail, \
                 float     update_scale, \
                 float     update_tail, \
                 float     output_zp, \
                 int       inner_size, \
                 int       axis_size, \
                 int       outer_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \
 \
    Image ref_i = create_image_from_image2d(ref, 2); \
    Image update_i = create_image_from_image2d(update, 2); \
    Image indices_i = create_image_from_image2d(indices, 4); \
    Image output_i = create_image_from_image2d(output, 2); \
 \
    dtype *ref_ptr = (dtype *)get_image_ptr_from_coord(ref_i, coord.xy); \
    dtype *output_ptr = (dtype *)get_image_ptr_from_coord(output_i, coord.xy); \
    dtype data = conv_func(convert_float(ref_ptr[0]) * ref_scale + ref_tail + output_zp); \
    if (coord.y < outer_size) \
    { \
        dtype *update_ptr = (dtype *)get_image_ptr_from_coord(update_i, coord.wy); \
        int *indices_ptr = (int *)get_image_ptr_from_coord(indices_i, coord.wy); \
        for(int x = 0; x < axis_size; x ++) \
        { \
            int offset = indices_ptr[x]; \
            if (offset == coord.x) \
            { \
                data += conv_func(convert_float(update_ptr[x]) * update_scale + update_tail + output_zp); \
            } \
        } \
    } \
 \
    output_ptr[0] = data; \
}
SE_ADD_AXIS0_16BITS_IMPL(I16_I32_I16toI16,    short,  convert_short_rte)
SE_ADD_AXIS0_16BITS_IMPL(F16_I32_F16toF16,    short,  convert_short)
SE_ADD_AXIS0_16BITS_IMPL(BF16_I32_BF16toBF16, ushort, convert_ushort)

#define SE_ADD_AXIS0_8BITS_IMPL(name, dtype, conv_func) \
__kernel void scatter_elements_add_axis0_##name \
    ( \
    __read_only  image2d_t ref, \
    __read_only  image2d_t indices, \
    __read_only  image2d_t update, \
    __write_only image2d_t output, \
                 int       axis, \
                 int       reduction, \
                 float     ref_scale, \
                 float     ref_tail, \
                 float     update_scale, \
                 float     update_tail, \
                 float     output_zp, \
                 int       inner_size, \
                 int       axis_size, \
                 int       outer_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \
 \
    Image ref_i = create_image_from_image2d(ref, 1); \
    Image update_i = create_image_from_image2d(update, 1); \
    Image indices_i = create_image_from_image2d(indices, 4); \
    Image output_i = create_image_from_image2d(output, 1); \
 \
    dtype *ref_ptr = (dtype *)get_image_ptr_from_coord(ref_i, coord.xy); \
    dtype *output_ptr = (dtype *)get_image_ptr_from_coord(output_i, coord.xy); \
    dtype data = conv_func(convert_float(ref_ptr[0]) * ref_scale + ref_tail + output_zp); \
    if (coord.y < outer_size) \
    { \
        dtype *update_ptr = (dtype *)get_image_ptr_from_coord(update_i, coord.wy); \
        int *indices_ptr = (int *)get_image_ptr_from_coord(indices_i, coord.wy); \
        for(int x = 0; x < axis_size; x ++) \
        { \
            int offset = indices_ptr[x]; \
            if (offset == coord.x) \
            { \
                data += conv_func(convert_float(update_ptr[x]) * update_scale + update_tail + output_zp); \
            } \
        } \
    } \
 \
    output_ptr[0] = data; \
}
SE_ADD_AXIS0_8BITS_IMPL(U8_I32_U8toU8, uchar, convert_uchar_rte)
SE_ADD_AXIS0_8BITS_IMPL(I8_I32_I8toI8, char,  convert_char)

#define SE_ADD_AXIS1_32BITS_IMPL(name, dtype) \
__kernel void scatter_elements_add_axis1_##name \
    ( \
    __read_only  image2d_array_t ref, \
    __read_only  image2d_array_t indices, \
    __read_only  image2d_array_t update, \
    __write_only image2d_array_t output, \
                 int             axis, \
                 int             reduction, \
                 float           ref_scale, \
                 float           ref_tail, \
                 float           update_scale, \
                 float           update_tail, \
                 float           output_zp, \
                 int             inner_size, \
                 int             axis_size, \
                 int             outer_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    Tensor ref_i = create_tensor_from_image2d_array(ref, 4); \
    Tensor update_i = create_tensor_from_image2d_array(update, 4); \
    Tensor indices_i = create_tensor_from_image2d_array(indices, 4); \
    Tensor output_i = create_tensor_from_image2d_array(output, 4); \
 \
    dtype *ref_ptr = (dtype *)get_tensor_ptr_from_coord(ref_i, coord); \
    dtype *output_ptr = (dtype *)get_tensor_ptr_from_coord(output_i, coord); \
    dtype data = ref_ptr[0]; \
    if (coord.x < inner_size && coord.z < outer_size) \
    { \
        dtype *update_ptr = (dtype *)get_tensor_ptr_from_coord(update_i, coord.xwzw); \
        int *indices_ptr = (int *)get_tensor_ptr_from_coord(indices_i, coord.xwzw); \
        for(int y = 0; y < axis_size; y ++) \
        { \
            int offset = indices_ptr[y * inner_size]; \
            if (offset == coord.y) \
            { \
                data += update_ptr[y * inner_size]; \
            } \
        } \
    } \
 \
    output_ptr[0] = data; \
}
SE_ADD_AXIS1_32BITS_IMPL(F32_I32_F32toF32, float)
SE_ADD_AXIS1_32BITS_IMPL(I32_I32_I32toI32, int)

#define SE_ADD_AXIS1_16BITS_IMPL(name, dtype, conv_func) \
__kernel void scatter_elements_add_axis1_##name \
    ( \
    __read_only  image2d_array_t ref, \
    __read_only  image2d_array_t indices, \
    __read_only  image2d_array_t update, \
    __write_only image2d_array_t output, \
                 int             axis, \
                 int             reduction, \
                 float           ref_scale, \
                 float           ref_tail, \
                 float           update_scale, \
                 float           update_tail, \
                 float           output_zp, \
                 int             inner_size, \
                 int             axis_size, \
                 int             outer_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    Tensor ref_i = create_tensor_from_image2d_array(ref, 2); \
    Tensor update_i = create_tensor_from_image2d_array(update, 2); \
    Tensor indices_i = create_tensor_from_image2d_array(indices, 4); \
    Tensor output_i = create_tensor_from_image2d_array(output, 2); \
 \
    dtype *ref_ptr = (dtype *)get_tensor_ptr_from_coord(ref_i, coord); \
    dtype *output_ptr = (dtype *)get_tensor_ptr_from_coord(output_i, coord); \
    dtype data = conv_func(convert_float(ref_ptr[0]) * ref_scale + ref_tail + output_zp); \
    if (coord.x < inner_size && coord.z < outer_size) \
    { \
        dtype *update_ptr = (dtype *)get_tensor_ptr_from_coord(update_i, coord.xwzw); \
        int *indices_ptr = (int *)get_tensor_ptr_from_coord(indices_i, coord.xwzw); \
        for(int y = 0; y < axis_size; y ++) \
        { \
            int offset = indices_ptr[y * inner_size]; \
            if (offset == coord.y) \
            { \
                data += conv_func(convert_float(update_ptr[y * inner_size]) \
                            * update_scale + update_tail + output_zp); \
            } \
        } \
    } \
 \
    output_ptr[0] = data; \
}
SE_ADD_AXIS1_16BITS_IMPL(I16_I32_I16toI16,    short,  convert_short_rte)
SE_ADD_AXIS1_16BITS_IMPL(F16_I32_F16toF16,    short,  convert_short)
SE_ADD_AXIS1_16BITS_IMPL(BF16_I32_BF16toBF16, ushort, convert_ushort)

#define SE_ADD_AXIS1_8BITS_IMPL(name, dtype, conv_func) \
__kernel void scatter_elements_add_axis1_##name \
    ( \
    __read_only  image2d_array_t ref, \
    __read_only  image2d_array_t indices, \
    __read_only  image2d_array_t update, \
    __write_only image2d_array_t output, \
                 int             axis, \
                 int             reduction, \
                 float           ref_scale, \
                 float           ref_tail, \
                 float           update_scale, \
                 float           update_tail, \
                 float           output_zp, \
                 int             inner_size, \
                 int             axis_size, \
                 int             outer_size \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
 \
    Tensor ref_i = create_tensor_from_image2d_array(ref, 1); \
    Tensor update_i = create_tensor_from_image2d_array(update, 1); \
    Tensor indices_i = create_tensor_from_image2d_array(indices, 4); \
    Tensor output_i = create_tensor_from_image2d_array(output, 1); \
 \
    dtype *ref_ptr = (dtype *)get_tensor_ptr_from_coord(ref_i, coord); \
    dtype *output_ptr = (dtype *)get_tensor_ptr_from_coord(output_i, coord); \
    dtype data = conv_func(convert_float(ref_ptr[0]) * ref_scale + ref_tail + output_zp); \
    if (coord.x < inner_size && coord.z < outer_size) \
    { \
        dtype *update_ptr = (dtype *)get_tensor_ptr_from_coord(update_i, coord.xwzw); \
        int *indices_ptr = (int *)get_tensor_ptr_from_coord(indices_i, coord.xwzw); \
        for(int y = 0; y < axis_size; y ++) \
        { \
            int offset = indices_ptr[y * inner_size]; \
            if (offset == coord.y) \
            { \
                data += conv_func(convert_float(update_ptr[y * inner_size]) \
                                * update_scale + update_tail + output_zp); \
            } \
        } \
    } \
 \
    output_ptr[0] = data; \
}
SE_ADD_AXIS1_8BITS_IMPL(U8_I32_U8toU8, uchar, convert_uchar_rte)
SE_ADD_AXIS1_8BITS_IMPL(I8_I32_I8toI8, char,  convert_char)
