__kernel void scatter_nd_U32toU32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    uint4 sum = (uint4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice = read_imagei(input0, (int2)(0, i));
        if(gidy == indice.x)
        {
            uint4 data = read_imageui(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imageui(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_U32toU32_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    uint4 sum = (uint4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice0 = read_imagei(input0, (int2)(0, i));
        int4 indice1 = read_imagei(input0, (int2)(1, i));
        int idx = indice0.x * width + indice1.x;
        if(gidy == idx)
        {
            uint4 data = read_imageui(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imageui(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_U32toU32_3D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    uint4 sum = (uint4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice0 = read_imagei(input0, (int2)(0, i));
        int4 indice1 = read_imagei(input0, (int2)(1, i));
        int4 indice2 = read_imagei(input0, (int2)(2, i));
        int idx = indice0.x * area + indice1.x * width + indice2.x;
        if(gidy == idx)
        {
            uint4 data = read_imageui(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imageui(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_I32toI32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 sum = (int4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice = read_imagei(input0, (int2)(0, i));
        if(gidy == indice.x)
        {
            int4 data = read_imagei(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imagei(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_I32toI32_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 sum = (int4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice0 = read_imagei(input0, (int2)(0, i));
        int4 indice1 = read_imagei(input0, (int2)(1, i));
        int idx = indice0.x * width + indice1.x;
        if(gidy == idx)
        {
            int4 data = read_imagei(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imagei(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_I32toI32_3D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    int4 sum = (int4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice0 = read_imagei(input0, (int2)(0, i));
        int4 indice1 = read_imagei(input0, (int2)(1, i));
        int4 indice2 = read_imagei(input0, (int2)(2, i));
        int idx = indice0.x * area + indice1.x * width + indice2.x;
        if(gidy == idx)
        {
            int4 data = read_imagei(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imagei(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_F32toF32_1D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    float4 sum = (float4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice = read_imagei(input0, (int2)(0, i));
        if(gidy == indice.x)
        {
            float4 data = read_imagef(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imagef(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_F32toF32_2D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    float4 sum = (float4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice0 = read_imagei(input0, (int2)(0, i));
        int4 indice1 = read_imagei(input0, (int2)(1, i));
        int idx = indice0.x * width + indice1.x;
        if(gidy == idx)
        {
            float4 data = read_imagef(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imagef(output, (int2)(gidx, gidy), sum);
}

__kernel void scatter_nd_F32toF32_3D(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int width,
    int area,
    int index_num
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num

    float4 sum = (float4)(0, 0, 0, 0);
    for(int i = 0; i < index_num; i++)
    {
        int4 indice0 = read_imagei(input0, (int2)(0, i));
        int4 indice1 = read_imagei(input0, (int2)(1, i));
        int4 indice2 = read_imagei(input0, (int2)(2, i));
        int idx = indice0.x * area + indice1.x * width + indice2.x;
        if(gidy == idx)
        {
            float4 data = read_imagef(input1, (int2)(gidx, i));
            sum += data;
        }
    }
    write_imagef(output, (int2)(gidx, gidy), sum);
}