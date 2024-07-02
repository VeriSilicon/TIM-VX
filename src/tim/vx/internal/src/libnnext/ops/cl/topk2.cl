
#define BITONIC_STEP(dtype) \
void bitonic_step_##dtype(uint num_stages, int lx, \
        __local dtype *local_data, __local int *local_indices) \
{ \
    for (uint stage = 0; stage < num_stages + 1; ++stage) \
    { \
        uint signo = (lx >> stage) & 1; \
 \
        for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            uint postShift = (stage - passOfStage); \
            uint pairDistance = 1 << postShift; \
 \
            uint left_id = ( (lx >> postShift) << (postShift + 1)) + (lx & (pairDistance - 1)); \
            uint right_id = left_id + pairDistance; \
 \
            int left_idx = local_indices[left_id]; \
            int right_idx = local_indices[right_id]; \
 \
            dtype left_elem = local_data[left_id]; \
            dtype right_elem = local_data[right_id]; \
 \
            if ((left_elem < right_elem || (left_elem == right_elem && left_idx < right_idx)) ^ signo) \
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
}
BITONIC_STEP(int)
BITONIC_STEP(uint)

#define BITONIC_STEP_ASCEND(dtype) \
void bitonic_step_ascend_##dtype(uint num_stages, int lx, \
        __local dtype *p_share_k, __local int *p_share_v) \
{ \
    for (uint stage = 0; stage < num_stages + 1; ++stage) \
    { \
        uint signo = (lx >> stage) & 1; \
 \
        for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
        { \
            uint postShift = (stage - passOfStage); \
            uint pairDistance = 1 << postShift; \
 \
            uint left_id = ( (lx >> postShift) << (postShift + 1)) + (lx & (pairDistance - 1)); \
            uint right_id = left_id + pairDistance; \
 \
            int left_idx = p_share_v[left_id]; \
            int right_idx = p_share_v[right_id]; \
 \
            dtype left_elem = p_share_k[left_id]; \
            dtype right_elem = p_share_k[right_id]; \
 \
            if ((left_elem > right_elem || (left_elem == right_elem && left_idx > right_idx)) ^ signo) \
            { \
                p_share_k[left_id] = right_elem; \
                p_share_k[right_id] = left_elem; \
 \
                p_share_v[left_id] = right_idx; \
                p_share_v[right_id] = left_idx; \
            } \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    } \
}
BITONIC_STEP_ASCEND(int)
BITONIC_STEP_ASCEND(uint)

#define BITONIC_MERGE(dtype) \
void bitonic_merge_##dtype(uint num_stages, int lx, \
        __local dtype *local_data, __local int *local_indices) \
{ \
    uint stage = num_stages; \
    uint signo = (lx >> stage) & 1; \
 \
    for (uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) \
    { \
        uint postShift = (stage - passOfStage); \
        uint pairDistance = 1 << postShift; \
 \
        uint left_id = ( (lx >> postShift) << (postShift + 1)) + (lx & (pairDistance - 1)); \
        uint right_id = left_id + pairDistance; \
 \
        int left_idx = local_indices[left_id]; \
        int right_idx = local_indices[right_id]; \
 \
        dtype left_elem = local_data[left_id]; \
        dtype right_elem = local_data[right_id]; \
 \
        if ((left_elem < right_elem || (left_elem == right_elem && left_idx < right_idx)) ^ signo) \
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
}
BITONIC_MERGE(int)
BITONIC_MERGE(uint)

#define BLOCK_SIZE              (512)

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1))) void topk_stage_I32toI32_I32
(
  __read_only  image2d_t input,
  __write_only image2d_t output,
  __write_only image2d_t indices,
               float     input_scale,
               float     input_tail,
               float     output_scale,
               float     output_tail,
               int       _num_stages,
               int       width
  )
 {
    uint lx = get_local_id(0);
    const int init_k = -2147483647;
    const int init_v = -2147483647;
    const int num_stages = 9;
    const int threads_per_block = BLOCK_SIZE;
    const int index_minus_1 = threads_per_block * 2 - 1;
    uint offset = 0;
    uint lx1 = lx + threads_per_block;

    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    __local int local_data[1536];
    __local int local_indices[1536];

    int left = read_imagei(input, coord.xy).x;
    coord.z += threads_per_block;
    int right = read_imagei(input, coord.zy).x;

    local_data[lx] = left;
    local_indices[lx] = coord.x;
    local_data[lx1] = right;
    local_indices[lx1] = coord.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    bitonic_step_int(num_stages, lx, local_data, local_indices);

    int min_data = local_data[511];

    int *p_share_k = local_data + threads_per_block;
    int *p_share_v = local_indices + threads_per_block;

    int limit = (width >> 10) << 10;
    p_share_k[lx] = init_k;
    p_share_v[lx] = init_v;

    p_share_k[lx1] = init_k;
    p_share_v[lx1] = init_v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (coord.x = lx + threads_per_block * 2; coord.x < limit; coord.x = coord.x + threads_per_block * 2)
    {
        int2 data;
        coord.z = coord.x + threads_per_block;
        data.x = read_imagei(input, coord.xy).x;
        data.y = read_imagei(input, coord.zy).x;

        p_share_k[lx] = data.x;
        p_share_v[lx] = coord.x;

        p_share_k[lx1] = data.y;
        p_share_v[lx1] = coord.z;
        barrier(CLK_LOCAL_MEM_FENCE);

        bitonic_step_ascend_int(num_stages, lx, p_share_k, p_share_v);

        if (p_share_k[index_minus_1] < min_data)
        {
            continue;
        }

        p_share_k[lx] = p_share_k[lx1];
        p_share_v[lx] = p_share_v[lx1];
        barrier(CLK_LOCAL_MEM_FENCE);

        bitonic_merge_int(num_stages, lx, local_data, local_indices);

        min_data = local_data[511];
        p_share_k[lx] = init_k;
        p_share_v[lx] = init_v;
        p_share_k[lx1] = init_k;
        p_share_v[lx1] = init_v;
    }

    if (width > limit)
    {
        if (coord.x < width)
        {
            int2 data;
            data.x = read_imagei(input, coord.xy).x;
            coord.z = coord.x + threads_per_block;
            data.y = read_imagei(input, coord.zy).x;

            p_share_k[lx] = data.x;
            p_share_v[lx] = coord.x;

            p_share_k[lx1] = coord.z < width ? data.y : init_k;
            p_share_v[lx1] = coord.z < width ? coord.z : init_v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        bitonic_step_ascend_int(num_stages, lx, p_share_k, p_share_v);

        if (p_share_k[index_minus_1] >= min_data)
        {
            p_share_k[lx] = p_share_k[lx1];
            p_share_v[lx] = p_share_v[lx1];
            barrier(CLK_LOCAL_MEM_FENCE);
            bitonic_merge_int(num_stages, lx, local_data, local_indices);
        }
    }

    int4 dst;
    dst.x = local_data[lx];

    coord.x = lx;
    write_imagei(output, coord.xy, dst.xxxx);

    int4 index;
    index.x = local_indices[lx];

    write_imagei(indices, coord.xy, index.xxxx);
}

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1))) void topk_stage_U32toU32_I32
(
  __read_only  image2d_t input,
  __write_only image2d_t output,
  __write_only image2d_t indices,
               float     input_scale,
               float     input_tail,
               float     output_scale,
               float     output_tail,
               int       _num_stages,
               int       width
  )
 {
    uint lx = get_local_id(0);
    const uint init_k = 0;
    const int init_v = -2147483647;
    const int num_stages = 9;
    const int threads_per_block = BLOCK_SIZE;
    const int index_minus_1 = threads_per_block * 2 - 1;
    uint offset = 0;
    uint lx1 = lx + threads_per_block;

    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    __local uint local_data[1536];
    __local int local_indices[1536];

    uint left = read_imageui(input, coord.xy).x;
    coord.z += threads_per_block;
    uint right = read_imageui(input, coord.zy).x;

    local_data[lx] = left;
    local_indices[lx] = coord.x;
    local_data[lx1] = right;
    local_indices[lx1] = coord.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    bitonic_step_uint(num_stages, lx, local_data, local_indices);

    uint min_data = local_data[511];

    uint *p_share_k = local_data + threads_per_block;
    int *p_share_v = local_indices + threads_per_block;

    int limit = (width >> 10) << 10;
    p_share_k[lx] = init_k;
    p_share_v[lx] = init_v;

    p_share_k[lx1] = init_k;
    p_share_v[lx1] = init_v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (coord.x = lx + threads_per_block * 2; coord.x < limit; coord.x = coord.x + threads_per_block * 2)
    {
        uint2 data;
        coord.z = coord.x + threads_per_block;
        data.x = read_imageui(input, coord.xy).x;
        data.y = read_imageui(input, coord.zy).x;

        p_share_k[lx] = data.x;
        p_share_v[lx] = coord.x;

        p_share_k[lx1] = data.y;
        p_share_v[lx1] = coord.z;
        barrier(CLK_LOCAL_MEM_FENCE);

        bitonic_step_ascend_uint(num_stages, lx, p_share_k, p_share_v);

        if (p_share_k[index_minus_1] < min_data)
        {
            continue;
        }

        p_share_k[lx] = p_share_k[lx1];
        p_share_v[lx] = p_share_v[lx1];
        barrier(CLK_LOCAL_MEM_FENCE);

        bitonic_merge_uint(num_stages, lx, local_data, local_indices);

        min_data = local_data[511];
        p_share_k[lx] = init_k;
        p_share_v[lx] = init_v;
        p_share_k[lx1] = init_k;
        p_share_v[lx1] = init_v;
    }

    if (width > limit)
    {
        if (coord.x < width)
        {
            uint2 data;
            data.x = read_imageui(input, coord.xy).x;
            coord.z = coord.x + threads_per_block;
            data.y = read_imageui(input, coord.zy).x;

            p_share_k[lx] = data.x;
            p_share_v[lx] = coord.x;

            p_share_k[lx1] = coord.z < width ? data.y : init_k;
            p_share_v[lx1] = coord.z < width ? coord.z : init_v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        bitonic_step_ascend_uint(num_stages, lx, p_share_k, p_share_v);

        if (p_share_k[index_minus_1] >= min_data)
        {
            p_share_k[lx] = p_share_k[lx1];
            p_share_v[lx] = p_share_v[lx1];
            barrier(CLK_LOCAL_MEM_FENCE);
            bitonic_merge_uint(num_stages, lx, local_data, local_indices);
        }
    }

    uint4 dst;
    dst.x = local_data[lx];

    coord.x = lx;
    write_imageui(output, coord.xy, dst.xxxx);

    int4 index;
    index.x = local_indices[lx];

    write_imagei(indices, coord.xy, index.xxxx);
}
