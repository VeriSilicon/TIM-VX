#define TOPK_F32(LOCAL_SIZE0, STAGES) \
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE0, 1, 1))) void topk_stage##STAGES##_F32toF32_I32 \
 ( \
  __read_only  image2d_t input, \
  __write_only image2d_t output, \
  __write_only image2d_t indices, \
               float     input_scale, \
               float     input_tail, \
               float     output_scale, \
               float     output_tail, \
               int       num_stages, \
               int       width \
  ) \
 { \
    uint local_id = get_local_id(0); \
    uint work_group_size = get_local_size(0); \
    uint offset = 0; \
 \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    __local float local_data[128]; \
    __local uint local_indices[128]; \
 \
    float left = read_imagef(input, coord.xy).x; \
    coord.z += work_group_size; \
    float data = read_imagef(input, coord.zy).x; \
    float right = coord.z < width ? data : -2147483647.0f; \
 \
    local_data[local_id] = left; \
    local_indices[local_id] = local_id; \
    local_data[local_id + work_group_size] = right; \
    local_indices[local_id + work_group_size] = local_id + work_group_size; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (uint stage = 0; stage < num_stages + 1; ++stage) \
    { \
        uint signo = (local_id >> stage) & 1; \
 \
        for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            uint postShift = (stage - passOfStage); \
            uint pairDistance = 1 << postShift; \
 \
            uint left_id = ( (local_id >> postShift) << (postShift + 1)) + (local_id & (pairDistance - 1)); \
            uint right_id = left_id + pairDistance; \
 \
            uint left_idx = local_indices[left_id]; \
            uint right_idx = local_indices[right_id]; \
 \
            float left_elem = local_data[left_id]; \
            float right_elem = local_data[right_id]; \
 \
            if ((left_elem < right_elem) ^ signo) \
            { \
                local_data[left_id] = right_elem; \
                local_data[right_id] = left_elem; \
 \
                local_indices[left_id] = right_idx; \
                local_indices[right_id] = left_idx; \
            } \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
 \
    float4 dst; \
    dst.x = local_data[local_id]; \
    dst.y = local_data[local_id + work_group_size]; \
 \
    write_imagef(output, coord.xy, dst.xxxx); \
    write_imagef(output, coord.zy, dst.yyyy); \
 \
    int4 index; \
    index.x = ((int*)local_indices)[local_id]; \
    index.y = ((int*)local_indices)[local_id + work_group_size]; \
 \
    write_imagei(indices, coord.xy, index.xxxx); \
    write_imagei(indices, coord.zy, index.yyyy); \
 }
TOPK_F32(1 << 0, 0)
TOPK_F32(1 << 1, 1)
TOPK_F32(1 << 2, 2)
TOPK_F32(1 << 3, 3)
TOPK_F32(1 << 4, 4)
TOPK_F32(1 << 5, 5)
TOPK_F32(1 << 6, 6)

#define TOPK_U32(LOCAL_SIZE0, STAGES) \
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE0, 1, 1))) void topk_stage##STAGES##_U32toU32_I32 \
 ( \
  __read_only  image2d_t input, \
  __write_only image2d_t output, \
  __write_only image2d_t indices, \
               float     input_scale, \
               float     input_tail, \
               float     output_scale, \
               float     output_tail, \
               int       num_stages, \
               int       width \
  ) \
 { \
    uint local_id = get_local_id(0); \
    uint work_group_size = get_local_size(0); \
    uint offset = 0; \
 \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    __local uint local_data[128]; \
    __local uint local_indices[128]; \
 \
    uint left = read_imageui(input, coord.xy).x; \
    coord.z += work_group_size; \
    uint data = read_imageui(input, coord.zy).x; \
    uint right = coord.z < width ? data : 0; \
 \
    local_data[local_id] = left; \
    local_indices[local_id] = local_id; \
    local_data[local_id + work_group_size] = right; \
    local_indices[local_id + work_group_size] = local_id + work_group_size; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (uint stage = 0; stage < num_stages + 1; ++stage) \
    { \
        uint signo = (local_id >> stage) & 1; \
 \
        for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            uint postShift = (stage - passOfStage); \
            uint pairDistance = 1 << postShift; \
 \
            uint left_id = ( (local_id >> postShift) << (postShift + 1)) + (local_id & (pairDistance - 1)); \
            uint right_id = left_id + pairDistance; \
 \
            uint left_idx = local_indices[left_id]; \
            uint right_idx = local_indices[right_id]; \
 \
            uint left_elem = local_data[left_id]; \
            uint right_elem = local_data[right_id]; \
 \
            if ((left_elem < right_elem) ^ signo) \
            { \
                local_data[left_id] = right_elem; \
                local_data[right_id] = left_elem; \
 \
                local_indices[left_id] = right_idx; \
                local_indices[right_id] = left_idx; \
            } \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
 \
    uint4 dst; \
    dst.x = local_data[local_id]; \
    dst.y = local_data[local_id + work_group_size]; \
 \
    write_imageui(output, coord.xy, dst.xxxx); \
    write_imageui(output, coord.zy, dst.yyyy); \
 \
    int4 index; \
    index.x = ((int*)local_indices)[local_id]; \
    index.y = ((int*)local_indices)[local_id + work_group_size]; \
 \
    write_imagei(indices, coord.xy, index.xxxx); \
    write_imagei(indices, coord.zy, index.yyyy); \
 }
TOPK_U32(1 << 0, 0)
TOPK_U32(1 << 1, 1)
TOPK_U32(1 << 2, 2)
TOPK_U32(1 << 3, 3)
TOPK_U32(1 << 4, 4)
TOPK_U32(1 << 5, 5)
TOPK_U32(1 << 6, 6)

#define TOPK_I32(LOCAL_SIZE0, STAGES) \
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE0, 1, 1))) void topk_stage##STAGES##_I32toI32_I32 \
 ( \
  __read_only  image2d_t input, \
  __write_only image2d_t output, \
  __write_only image2d_t indices, \
               float     input_scale, \
               float     input_tail, \
               float     output_scale, \
               float     output_tail, \
               int       num_stages, \
               int       width \
  ) \
 { \
    int local_id = get_local_id(0); \
    int work_group_size = get_local_size(0); \
    int offset = 0; \
 \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    __local int local_data[128]; \
    __local int local_indices[128]; \
 \
    int left = read_imagei(input, coord.xy).x; \
    coord.z += work_group_size; \
    int data = read_imagei(input, coord.zy).x; \
    int right = coord.z < width ? data : -2147483647; \
 \
    local_data[local_id] = left; \
    local_indices[local_id] = local_id; \
    local_data[local_id + work_group_size] = right; \
    local_indices[local_id + work_group_size] = local_id + work_group_size; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (int stage = 0; stage < num_stages + 1; ++stage) \
    { \
        int signo = (local_id >> stage) & 1; \
 \
        for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            int postShift = (stage - passOfStage); \
            int pairDistance = 1 << postShift; \
 \
            int left_id = ( (local_id >> postShift) << (postShift + 1)) + (local_id & (pairDistance - 1)); \
            int right_id = left_id + pairDistance; \
 \
            int left_idx = local_indices[left_id]; \
            int right_idx = local_indices[right_id]; \
 \
            int left_elem = local_data[left_id]; \
            int right_elem = local_data[right_id]; \
 \
            if ((left_elem < right_elem) ^ signo) \
            { \
                local_data[left_id] = right_elem; \
                local_data[right_id] = left_elem; \
 \
                local_indices[left_id] = right_idx; \
                local_indices[right_id] = left_idx; \
            } \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
 \
    int4 dst; \
    dst.x = local_data[local_id]; \
    dst.y = local_data[local_id + work_group_size]; \
 \
    write_imagei(output, coord.xy, dst.xxxx); \
    write_imagei(output, coord.zy, dst.yyyy); \
 \
    int4 index; \
    index.x = ((int*)local_indices)[local_id]; \
    index.y = ((int*)local_indices)[local_id + work_group_size]; \
 \
    write_imagei(indices, coord.xy, index.xxxx); \
    write_imagei(indices, coord.zy, index.yyyy); \
 }
TOPK_I32(1 << 0, 0)
TOPK_I32(1 << 1, 1)
TOPK_I32(1 << 2, 2)
TOPK_I32(1 << 3, 3)
TOPK_I32(1 << 4, 4)
TOPK_I32(1 << 5, 5)
TOPK_I32(1 << 6, 6)

#define TOPK_F32toU32(LOCAL_SIZE0, STAGES) \
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE0, 1, 1))) void topk_stage##STAGES##_F32toU32_I32 \
 ( \
  __read_only  image2d_t input, \
  __write_only image2d_t output, \
  __write_only image2d_t indices, \
               float     input_scale, \
               float     input_tail, \
               float     output_scale, \
               float     output_tail, \
               int       num_stages, \
               int       width \
  ) \
 { \
    uint local_id = get_local_id(0); \
    uint work_group_size = get_local_size(0); \
    uint offset = 0; \
 \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    __local float local_data[128]; \
    __local uint local_indices[128]; \
 \
    float left = read_imagef(input, coord.xy).x; \
    coord.z += work_group_size; \
    float data = read_imagef(input, coord.zy).x; \
    float right = coord.z < width ? data : -2147483647.0f; \
 \
    local_data[local_id] = left; \
    local_indices[local_id] = local_id; \
    local_data[local_id + work_group_size] = right; \
    local_indices[local_id + work_group_size] = local_id + work_group_size; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (uint stage = 0; stage < num_stages + 1; ++stage) \
    { \
        uint signo = (local_id >> stage) & 1; \
 \
        for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            uint postShift = (stage - passOfStage); \
            uint pairDistance = 1 << postShift; \
 \
            uint left_id = ( (local_id >> postShift) << (postShift + 1)) + (local_id & (pairDistance - 1)); \
            uint right_id = left_id + pairDistance; \
 \
            uint left_idx = local_indices[left_id]; \
            uint right_idx = local_indices[right_id]; \
 \
            float left_elem = local_data[left_id]; \
            float right_elem = local_data[right_id]; \
 \
            if ((left_elem < right_elem) ^ signo) \
            { \
                local_data[left_id] = right_elem; \
                local_data[right_id] = left_elem; \
 \
                local_indices[left_id] = right_idx; \
                local_indices[right_id] = left_idx; \
            } \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
 \
    uint4 dst; \
    dst.x = convert_uint(local_data[local_id] * output_scale + output_tail); \
    dst.y = convert_uint(local_data[local_id + work_group_size] * output_scale + output_tail); \
    write_imageui(output, coord.xy, dst.xxxx); \
    write_imageui(output, coord.zy, dst.yyyy); \
 \
    int4 index; \
    index.x = ((int*)local_indices)[local_id]; \
    index.y = ((int*)local_indices)[local_id + work_group_size]; \
 \
    write_imagei(indices, coord.xy, index.xxxx); \
    write_imagei(indices, coord.zy, index.yyyy); \
 }

TOPK_F32toU32(1 << 0, 0)
TOPK_F32toU32(1 << 1, 1)
TOPK_F32toU32(1 << 2, 2)
TOPK_F32toU32(1 << 3, 3)
TOPK_F32toU32(1 << 4, 4)
TOPK_F32toU32(1 << 5, 5)
TOPK_F32toU32(1 << 6, 6)

#define TOPK_F32toI32(LOCAL_SIZE0, STAGES) \
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE0, 1, 1))) void topk_stage##STAGES##_F32toI32_I32 \
 ( \
  __read_only  image2d_t input, \
  __write_only image2d_t output, \
  __write_only image2d_t indices, \
               float     input_scale, \
               float     input_tail, \
               float     output_scale, \
               float     output_tail, \
               int       num_stages, \
               int       width \
  ) \
 { \
    uint local_id = get_local_id(0); \
    uint work_group_size = get_local_size(0); \
    uint offset = 0; \
 \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1)); \
 \
    __local float local_data[128]; \
    __local uint local_indices[128]; \
 \
    float left = read_imagef(input, coord.xy).x; \
    coord.z += work_group_size; \
    float data = read_imagef(input, coord.zy).x; \
    float right = coord.z < width ? data : -2147483647.0f; \
 \
    local_data[local_id] = left; \
    local_indices[local_id] = local_id; \
    local_data[local_id + work_group_size] = right; \
    local_indices[local_id + work_group_size] = local_id + work_group_size; \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    for (uint stage = 0; stage < num_stages + 1; ++stage) \
    { \
        uint signo = (local_id >> stage) & 1; \
 \
        for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            uint postShift = (stage - passOfStage); \
            uint pairDistance = 1 << postShift; \
 \
            uint left_id = ( (local_id >> postShift) << (postShift + 1)) + (local_id & (pairDistance - 1)); \
            uint right_id = left_id + pairDistance; \
 \
            uint left_idx = local_indices[left_id]; \
            uint right_idx = local_indices[right_id]; \
 \
            float left_elem = local_data[left_id]; \
            float right_elem = local_data[right_id]; \
 \
            if ((left_elem < right_elem) ^ signo) \
            { \
                local_data[left_id] = right_elem; \
                local_data[right_id] = left_elem; \
 \
                local_indices[left_id] = right_idx; \
                local_indices[right_id] = left_idx; \
            } \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
 \
    int4 dst; \
    dst.x = convert_int(local_data[local_id] * output_scale + output_tail); \
    dst.y = convert_int(local_data[local_id + work_group_size] * output_scale + output_tail); \
    write_imagei(output, coord.xy, dst.xxxx); \
    write_imagei(output, coord.zy, dst.yyyy); \
 \
    int4 index; \
    index.x = ((int*)local_indices)[local_id]; \
    index.y = ((int*)local_indices)[local_id + work_group_size]; \
 \
    write_imagei(indices, coord.xy, index.xxxx); \
    write_imagei(indices, coord.zy, index.yyyy); \
 }

TOPK_F32toI32(1 << 0, 0)
TOPK_F32toI32(1 << 1, 1)
TOPK_F32toI32(1 << 2, 2)
TOPK_F32toI32(1 << 3, 3)
TOPK_F32toI32(1 << 4, 4)
TOPK_F32toI32(1 << 5, 5)
TOPK_F32toI32(1 << 6, 6)