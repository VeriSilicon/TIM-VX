
inline void AtomicAdd_float(volatile __global float *source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do
    {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while(atomic_cmpxchg((volatile __global unsigned int *)source,
                             prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void AtomicMul_float(volatile __global float *source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do
    {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal * operand;
    } while(atomic_cmpxchg((volatile __global unsigned int *)source,
                             prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void AtomicMax_float(volatile __global float *source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do
    {
        prevVal.floatVal = *source;
        newVal.floatVal = fmax(prevVal.floatVal, operand);
    } while(atomic_cmpxchg((volatile __global unsigned int *)source,
                             prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void AtomicMin_float(volatile __global float *source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do
    {
        prevVal.floatVal = *source;
        newVal.floatVal = fmin(prevVal.floatVal, operand);
    } while(atomic_cmpxchg((volatile __global unsigned int *)source,
                             prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

#define SCATTER_REDUCTION_PREPROCESS(name0, ptr0, type0, size0, ptr2) \
__kernel void scatter_nd_update_reduction_preprocess_##name0( \
    __read_only image2d_t   input_ref, \
    image2d_t  temp_buf_float, \
    int length, int res, float input_scale, float zp_scale) \
{ \
    int gidx = get_global_id(0); \
    Image img1 = create_image_from_image2d(input_ref, size0); \
    Image img2 = create_image_from_image2d(temp_buf_float, 4); \
    __global float* tmp_ref_ptr = (__global float*)img2.ptr; \
    type0 src0, src1; \
    float4 tmpDst0, tmpDst1; \
    __global ptr2* input_ptr = (__global ptr2*)img1.ptr; \
    if(length > 0) \
    { \
        int loc2 = gidx * 8; \
        ptr0 tmpData0 = vload4(0, input_ptr + loc2); \
        ptr0 tmpData1 = vload4(1, input_ptr + loc2); \
        _viv_asm(COPY, src0, tmpData0, 16); \
        _viv_asm(COPY, src1, tmpData1, 16); \
        _viv_asm(CONV, tmpDst0, src0); \
        _viv_asm(CONV, tmpDst1, src1); \
        tmpDst0 = tmpDst0 * input_scale + zp_scale; \
        tmpDst1 = tmpDst1 * input_scale + zp_scale; \
        vstore4(tmpDst0, 0, tmp_ref_ptr + loc2); \
        vstore4(tmpDst1, 1, tmp_ref_ptr + loc2); \
    } \
    for(int i = gidx; i < res; i += get_global_size(0)) \
    { \
        ptr2 tmpData0 = input_ptr[length + i]; \
        _viv_asm(COPY, src0, tmpData0, 4); \
        _viv_asm(CONV, tmpDst0, src0); \
        tmpDst0.x = tmpDst0.x * input_scale + zp_scale; \
        tmp_ref_ptr[length + i] = tmpDst0.x; \
    } \
}
SCATTER_REDUCTION_PREPROCESS(U8,  uchar4, uchar4, 1, uchar)
SCATTER_REDUCTION_PREPROCESS(I8,  char4,  char4,  1, char)
SCATTER_REDUCTION_PREPROCESS(I16, short4, short4, 2, short)
SCATTER_REDUCTION_PREPROCESS(F16, short4, half4,  2, short)
SCATTER_REDUCTION_PREPROCESS(F32, float4, float4, 4, float)

#define SCATTER_ND_REDUCTION_PROCESS_F16(name0, func) \
__kernel void scatter_nd_update_reduction_##name0##_F16( \
    __read_only image2d_t   index, \
    __read_only image2d_t   update, \
    image2d_t  temp_buf_float, \
    image2d_t  link_buffer0, \
    int val0, int val1, int val2, int val3, int val4, int val5, int val6, \
    int coord_dim, int update_width, int output_width, float update_scale, float zp_scale) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    Image img1 = create_image_from_image2d(index, 4); \
    Image img2 = create_image_from_image2d(update, 2); \
    Image img3 = create_image_from_image2d(temp_buf_float, 4); \
    __global int* index_ptr = (__global int*)img1.ptr; \
    __global short* update_ptr = (__global short*)img2.ptr; \
    __global float* output_ptr = (__global float*)img3.ptr; \
    half src; \
 \
    int4 indice = vload4(0, index_ptr + gidy * coord_dim); \
    int4 indice1 = coord_dim < 5 ? (int4)(0) : vload4(1, index_ptr + gidy * coord_dim); \
    short tmpData = update_ptr[gidy * update_width + gidx]; \
    int idx = indice.x * val0 + indice.y * val1 + indice.z * val2 + indice.w * val3; \
    idx = idx + indice1.x * val4 + indice1.y * val5 + indice1.z * val6; \
    int loc = idx * output_width + gidx; \
    _viv_asm(COPY, src, tmpData, 4); \
    float data; \
    _viv_asm(CONV, data, src); \
    func(output_ptr + loc, data); \
}
SCATTER_ND_REDUCTION_PROCESS_F16(Add,  AtomicAdd_float)
SCATTER_ND_REDUCTION_PROCESS_F16(Mul,  AtomicMul_float)
SCATTER_ND_REDUCTION_PROCESS_F16(Max,  AtomicMax_float)
SCATTER_ND_REDUCTION_PROCESS_F16(Min,  AtomicMin_float)

#define SCATTER_ND_UPDATE_PROCESS_QINT(name0, src0_type, ptr_type, element_size, func) \
__kernel void scatter_nd_update_reduction_##name0##_##src0_type( \
    __read_only image2d_t   index, \
    __read_only image2d_t   update, \
    image2d_t  temp_buf_float, \
    image2d_t  link_buffer0, \
    int val0, int val1, int val2, int val3, int val4, int val5, int val6, \
    int coord_dim, int update_width, int output_width, float update_scale, float zp_scale) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
    Image img1 = create_image_from_image2d(index, 4); \
    Image img2 = create_image_from_image2d(update, element_size); \
    Image img3 = create_image_from_image2d(temp_buf_float, 4); \
    __global int* index_ptr = (__global int*)img1.ptr; \
    __global ptr_type* update_ptr = (__global ptr_type*)img2.ptr; \
    __global float* output_ptr = (__global float*)img3.ptr; \
 \
    int4 indice = vload4(0, index_ptr + gidy * coord_dim); \
    int4 indice1 = coord_dim < 5 ? (int4)(0) : vload4(1, index_ptr + gidy * coord_dim); \
    ptr_type tmpData = update_ptr[gidy * update_width + gidx]; \
    int idx = indice.x * val0 + indice.y * val1 + indice.z * val2 + indice.w * val3; \
    idx = idx + indice1.x * val4 + indice1.y * val5 + indice1.z * val6; \
    int loc = idx * output_width + gidx; \
    float data; \
    _viv_asm(CONV, data, tmpData); \
    data = data * update_scale + zp_scale; \
    func(output_ptr + loc, data); \
}
SCATTER_ND_UPDATE_PROCESS_QINT(Add, U8,  uchar, 1, AtomicAdd_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Mul, U8,  uchar, 1, AtomicMul_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Max, U8,  uchar, 1, AtomicMax_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Min, U8,  uchar, 1, AtomicMin_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Add, I8,  char,  1, AtomicAdd_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Mul, I8,  char,  1, AtomicMul_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Max, I8,  char,  1, AtomicMax_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Min, I8,  char,  1, AtomicMin_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Add, I16, short, 2, AtomicAdd_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Mul, I16, short, 2, AtomicMul_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Max, I16, short, 2, AtomicMax_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Min, I16, short, 2, AtomicMin_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Add, F32, float, 4, AtomicAdd_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Mul, F32, float, 4, AtomicMul_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Max, F32, float, 4, AtomicMax_float)
SCATTER_ND_UPDATE_PROCESS_QINT(Min, F32, float, 4, AtomicMin_float)