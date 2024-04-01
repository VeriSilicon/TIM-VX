#define LOCAL_SIZE_X    (32)
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1))) void topk_odd_even_sort_F32toF32_I32
 (
  __read_only  image2d_t input,
               image2d_t input_t,
               image2d_t indices_t,
  __write_only image2d_t output,
  __write_only image2d_t indices,
               float     input_scale,
               float     input_tail,
               float     output_scale,
               float     output_tail,
               int       width
  )
 {
    uint lid = get_local_id(0);
    uint work_group_size = get_local_size(0);
    uint offset = 0;

    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    for (coord.x = lid; coord.x < width; coord.x += LOCAL_SIZE_X)
    {
        float4 data = read_imagef(input, coord.xy);

        write_imagef(input_t, coord.xy, data);
        write_imagei(indices_t, coord.xy, coord.xxxx);
    }

    __local int sorted[1];

    sorted[0] = 0;

    while (1)
    {
        if (lid == 0)
        {
            *sorted = 0;
        }
        int swapped = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        // odd-even
        coord.x = lid * 2;
        coord.z = lid * 2 + 1;
        for (; coord.z < width; )
        {
            float4 left = read_imagef(input_t, coord.xy);
            float4 right = read_imagef(input_t, coord.zy);
            int4 l_index = read_imagei(indices_t, coord.xy);
            int4 r_index = read_imagei(indices_t, coord.zy);

            if ( (left.x < right.x) ||
                (left.x == right.x && l_index.x < r_index.x) )
            {
                swapped = 1;

                write_imagef(input_t, coord.xy, right);
                write_imagef(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz += 2 * LOCAL_SIZE_X;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        // even-odd
        coord.x = lid * 2 + 1;
        coord.z = lid * 2 + 2;
        for (; coord.z < width; )
        {
            float4 left = read_imagef(input_t, coord.xy);
            float4 right = read_imagef(input_t, coord.zy);
            int4 l_index = read_imagei(indices_t, coord.xy);
            int4 r_index = read_imagei(indices_t, coord.zy);

            if ( (left.x < right.x) ||
                (left.x == right.x && l_index.x < r_index.x) )
            {
                swapped = 1;

                write_imagef(input_t, coord.xy, right);
                write_imagef(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz += 2 * LOCAL_SIZE_X;
        }

        atomic_add(sorted, swapped);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (*sorted == 0)
            break;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (coord.x = lid; coord.x < width; coord.x += LOCAL_SIZE_X)
    {
        float4 data = read_imagef(input_t, coord.xy);
        int4 index = read_imagei(indices_t, coord.xy);

        write_imagef(output, coord.xy, data);
        write_imagei(indices, coord.xy, index);
    }
}

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1))) void topk_odd_even_sort_U32toU32_I32
 (
  __read_only  image2d_t input,
               image2d_t input_t,
               image2d_t indices_t,
  __write_only image2d_t output,
  __write_only image2d_t indices,
               float     input_scale,
               float     input_tail,
               float     output_scale,
               float     output_tail,
               int       width
  )
 {
    uint lid = get_local_id(0);
    uint work_group_size = get_local_size(0);
    uint offset = 0;

    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    for (coord.x = lid; coord.x < width; coord.x += LOCAL_SIZE_X)
    {
        uint4 data = read_imageui(input, coord.xy);

        write_imageui(input_t, coord.xy, data);
        write_imagei(indices_t, coord.xy, coord.xxxx);
    }

    __local int sorted[1];
    sorted[0] = 0;

    while (1)
    {
        if (lid == 0)
        {
            *sorted = 0;
        }
        int swapped = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        // odd-even
        coord.x = lid * 2;
        coord.z = lid * 2 + 1;
        for (; coord.z < width; )
        {
            uint4 left = read_imageui(input_t, coord.xy);
            uint4 right = read_imageui(input_t, coord.zy);
            int4 l_index = read_imagei(indices_t, coord.xy);
            int4 r_index = read_imagei(indices_t, coord.zy);

            if ( (left.x < right.x) ||
                (left.x == right.x && l_index.x < r_index.x) )
            {
                swapped = 1;

                write_imageui(input_t, coord.xy, right);
                write_imageui(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz += 2 * LOCAL_SIZE_X;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        // even-odd
        coord.x = lid * 2 + 1;
        coord.z = lid * 2 + 2;
        for (; coord.z < width; )
        {
            uint4 left = read_imageui(input_t, coord.xy);
            uint4 right = read_imageui(input_t, coord.zy);
            int4 l_index = read_imagei(indices_t, coord.xy);
            int4 r_index = read_imagei(indices_t, coord.zy);

            if ( (left.x < right.x) ||
                (left.x == right.x && l_index.x < r_index.x) )
            {
                swapped = 1;

                write_imageui(input_t, coord.xy, right);
                write_imageui(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz += 2 * LOCAL_SIZE_X;
        }

        atomic_add(sorted, swapped);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (*sorted == 0)
            break;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (coord.x = lid; coord.x < width; coord.x += LOCAL_SIZE_X)
    {
        uint4 data = read_imageui(input_t, coord.xy);
        int4 index = read_imagei(indices_t, coord.xy);

        write_imageui(output, coord.xy, data);
        write_imagei(indices, coord.xy, index);
    }
}

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1))) void topk_odd_even_sort_I32toI32_I32
 (
  __read_only  image2d_t input,
               image2d_t input_t,
               image2d_t indices_t,
  __write_only image2d_t output,
  __write_only image2d_t indices,
               float     input_scale,
               float     input_tail,
               float     output_scale,
               float     output_tail,
               int       width
  )
 {
    uint lid = get_local_id(0);
    uint work_group_size = get_local_size(0);
    uint offset = 0;

    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));

    for (coord.x = lid; coord.x < width; coord.x += LOCAL_SIZE_X)
    {
        int4 data = read_imagei(input, coord.xy);

        write_imagei(input_t, coord.xy, data);
        write_imagei(indices_t, coord.xy, coord.xxxx);
    }

    __local int sorted[1];
    sorted[0] = 0;

    while (1)
    {
        if (lid == 0)
        {
            *sorted = 0;
        }
        int swapped = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        // odd-even
        coord.x = lid * 2;
        coord.z = lid * 2 + 1;
        for (; coord.z < width; )
        {
            int4 left = read_imagei(input_t, coord.xy);
            int4 right = read_imagei(input_t, coord.zy);
            int4 l_index = read_imagei(indices_t, coord.xy);
            int4 r_index = read_imagei(indices_t, coord.zy);

            if ( (left.x < right.x) ||
                (left.x == right.x && l_index.x < r_index.x) )
            {
                swapped = 1;

                write_imagei(input_t, coord.xy, right);
                write_imagei(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz += 2 * LOCAL_SIZE_X;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        // even-odd
        coord.x = lid * 2 + 1;
        coord.z = lid * 2 + 2;
        for (; coord.z < width; )
        {
            int4 left = read_imagei(input_t, coord.xy);
            int4 right = read_imagei(input_t, coord.zy);
            int4 l_index = read_imagei(indices_t, coord.xy);
            int4 r_index = read_imagei(indices_t, coord.zy);

            if ( (left.x < right.x) ||
                (left.x == right.x && l_index.x < r_index.x) )
            {
                swapped = 1;

                write_imagei(input_t, coord.xy, right);
                write_imagei(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz += 2 * LOCAL_SIZE_X;
        }

        atomic_add(sorted, swapped);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (*sorted == 0)
            break;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (coord.x = lid; coord.x < width; coord.x += LOCAL_SIZE_X)
    {
        int4 data = read_imagei(input_t, coord.xy);
        int4 index = read_imagei(indices_t, coord.xy);

        write_imagei(output, coord.xy, data);
        write_imagei(indices, coord.xy, index);
    }
}
