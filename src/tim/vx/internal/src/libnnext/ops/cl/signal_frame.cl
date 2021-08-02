
#define SIGNAL_FRAME_SH_IMPL(type, data_type, read_imagefunc, write_imagefunc) \
__kernel void signal_frame_##type##to##type \
    ( \
    __read_only  image2d_t       input, \
    __write_only image2d_array_t output, \
                 int             frame_step \
    ) \
{ \
    int inner = get_global_id(0); \
    int length_k = get_global_id(1); \
    int frames_id = get_global_id(2); \
 \
    int4 coord = (int4)(inner, length_k, frames_id, frames_id); \
    int2 coord_in = (int2)(inner, frames_id * frame_step + length_k); \
 \
    data_type src = read_imagefunc(input, coord_in); \
    write_imagefunc(output, coord, src); \
}
SIGNAL_FRAME_SH_IMPL(F32, float4, read_imagef,  write_imagef)
SIGNAL_FRAME_SH_IMPL(U8,  uint4,  read_imageui, write_imageui)
