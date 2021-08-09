
#define SCATTER_ND_UPDATE(src0_type, data_type, read_func, write_func) \
__kernel void scatter_nd_update_##src0_type##src0_type##to##src0_type( \
    __read_only image2d_t   input0, \
    __read_only image2d_t   input1, \
    __read_only image2d_t   input2, \
    __write_only image2d_t  output, \
    int offsetX, \
    int offsetY, \
    int offsetZ, \
    int offsetW, \
    int offset_idx, \
    int coord_dim, \
    int index_num \
    ) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    int cnt = 0; \
 \
    data_type sum = (data_type)(0, 0, 0, 0); \
    Image img1 = create_image_from_image2d(input1, 4); \
    __global int* index_ptr = (__global int*)img1.ptr; \
    for(int i = 0; i < index_num; i++) \
    { \
        int4 indice = vload4(0, index_ptr + offset_idx); \
        index_ptr += coord_dim; \
        int idx = indice.x * offsetX + indice.y * offsetY + indice.z * offsetZ + indice.w * offsetW; \
        if(gidy == idx) \
        { \
            data_type data = read_func(input2, (int2)(gidx, i)); \
            cnt++; \
            sum += data; \
        } \
    } \
    int2 coord = (int2)(gidx, gidy); \
    if(cnt == 0) \
    { \
        sum = read_func(input0, coord); \
    } \
    write_func(output, coord, sum); \
}
SCATTER_ND_UPDATE(U32,  uint4,  read_imageui, write_imageui)
SCATTER_ND_UPDATE(I32,  int4,   read_imagei,  write_imagei)
SCATTER_ND_UPDATE(F32,  float4, read_imagef,  write_imagef)
