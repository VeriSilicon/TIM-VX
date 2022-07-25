#define LOCAL_SIZE_X    (32)
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1))) void topk_odd_even_sort_F32toF32_I32
 (
  __read_only  image2d_t input,
               image2d_t input_t,
               image2d_t indices_t,
  __write_only image2d_t output,
  __write_only image2d_t indices,
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
    int width_minus_one = width - 1;
    int num_pixels_per_thread = (width_minus_one + LOCAL_SIZE_X) / LOCAL_SIZE_X;
    num_pixels_per_thread = num_pixels_per_thread + (num_pixels_per_thread & 1);

    int x_start = lid * num_pixels_per_thread;
    int x_end = min(lid * num_pixels_per_thread + num_pixels_per_thread, width_minus_one);

    sorted[0] = 0;

    while (1)
    {
        if (lid == 0)
        {
            *sorted = 0;
        }
        int swapped = 0;
        barrier(CLK_GLOBAL_MEM_FENCE);

        // odd-even
        coord.x = x_start;
        coord.z = x_start + 1;
        for (; coord.x < x_end; )
        {
            float4 left = read_imagef(input_t, coord.xy);
            float4 right = read_imagef(input_t, coord.zy);

            if (left.x < right.x)
            {
                int4 l_index = read_imagei(indices_t, coord.xy);
                int4 r_index = read_imagei(indices_t, coord.zy);
                swapped = 1;

                write_imagef(input_t, coord.xy, right);
                write_imagef(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz = coord.xz + 2;
        }

        // even-odd
        coord.x = x_start + 1;
        coord.z = x_start + 2;
        for (; coord.x < x_end; )
        {
            float4 left = read_imagef(input_t, coord.xy);
            float4 right = read_imagef(input_t, coord.zy);

            if (left.x < right.x)
            {
                int4 l_index = read_imagei(indices_t, coord.xy);
                int4 r_index = read_imagei(indices_t, coord.zy);
                swapped = 1;

                write_imagef(input_t, coord.xy, right);
                write_imagef(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz = coord.xz + 2;
        }

        atomic_add(sorted, swapped);
        barrier(CLK_GLOBAL_MEM_FENCE);

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
    int width_minus_one = width - 1;
    int num_pixels_per_thread = (width_minus_one + LOCAL_SIZE_X) / LOCAL_SIZE_X;
    num_pixels_per_thread = num_pixels_per_thread + (num_pixels_per_thread & 1);

    int x_start = lid * num_pixels_per_thread;
    int x_end = min(lid * num_pixels_per_thread + num_pixels_per_thread, width_minus_one);

    sorted[0] = 0;

    while (1)
    {
        if (lid == 0)
        {
            *sorted = 0;
        }
        int swapped = 0;
        barrier(CLK_GLOBAL_MEM_FENCE);

        // odd-even
        coord.x = x_start;
        coord.z = x_start + 1;
        for (; coord.x < x_end; )
        {
            uint4 left = read_imageui(input_t, coord.xy);
            uint4 right = read_imageui(input_t, coord.zy);

            if (left.x < right.x)
            {
                int4 l_index = read_imagei(indices_t, coord.xy);
                int4 r_index = read_imagei(indices_t, coord.zy);
                swapped = 1;

                write_imageui(input_t, coord.xy, right);
                write_imageui(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz = coord.xz + 2;
        }

        // even-odd
        coord.x = x_start + 1;
        coord.z = x_start + 2;
        for (; coord.x < x_end; )
        {
            uint4 left = read_imageui(input_t, coord.xy);
            uint4 right = read_imageui(input_t, coord.zy);

            if (left.x < right.x)
            {
                int4 l_index = read_imagei(indices_t, coord.xy);
                int4 r_index = read_imagei(indices_t, coord.zy);
                swapped = 1;

                write_imageui(input_t, coord.xy, right);
                write_imageui(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz = coord.xz + 2;
        }

        atomic_add(sorted, swapped);
        barrier(CLK_GLOBAL_MEM_FENCE);

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
    int width_minus_one = width - 1;
    int num_pixels_per_thread = (width_minus_one + LOCAL_SIZE_X) / LOCAL_SIZE_X;
    num_pixels_per_thread = num_pixels_per_thread + (num_pixels_per_thread & 1);

    int x_start = lid * num_pixels_per_thread;
    int x_end = min(lid * num_pixels_per_thread + num_pixels_per_thread, width_minus_one);

    sorted[0] = 0;

    while (1)
    {
        if (lid == 0)
        {
            *sorted = 0;
        }
        int swapped = 0;
        barrier(CLK_GLOBAL_MEM_FENCE);

        // odd-even
        coord.x = x_start;
        coord.z = x_start + 1;
        for (; coord.x < x_end; )
        {
            int4 left = read_imagei(input_t, coord.xy);
            int4 right = read_imagei(input_t, coord.zy);

            if (left.x < right.x)
            {
                int4 l_index = read_imagei(indices_t, coord.xy);
                int4 r_index = read_imagei(indices_t, coord.zy);
                swapped = 1;

                write_imagei(input_t, coord.xy, right);
                write_imagei(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz = coord.xz + 2;
        }

        // even-odd
        coord.x = x_start + 1;
        coord.z = x_start + 2;
        for (; coord.x < x_end; )
        {
            int4 left = read_imagei(input_t, coord.xy);
            int4 right = read_imagei(input_t, coord.zy);

            if (left.x < right.x)
            {
                int4 l_index = read_imagei(indices_t, coord.xy);
                int4 r_index = read_imagei(indices_t, coord.zy);
                swapped = 1;

                write_imagei(input_t, coord.xy, right);
                write_imagei(input_t, coord.zy, left);

                write_imagei(indices_t, coord.xy, r_index);
                write_imagei(indices_t, coord.zy, l_index);
            }

            coord.xz = coord.xz + 2;
        }

        atomic_add(sorted, swapped);
        barrier(CLK_GLOBAL_MEM_FENCE);

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