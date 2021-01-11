__kernel void gemm_F32F32toF32_2D(
    __read_only image2d_t   inputA,
    __read_only image2d_t   inputB,
    __write_only image2d_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int2 coord_a = (int2)(0, gidy);
    int2 coord_b = (int2)(gidx, 0);

    float4 sum = (float4)(0);

    for(; coord_a.x < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord_a);
        tempB0 = read_imagef(inputB, coord_b);
        coord_a.x++;
        coord_b.y++;

        sum += tempA0 * tempB0;
    }

    coord_b.y = gidy;
    write_imagef(output, coord_b, sum);
}

__kernel void gemm_F32F32toF32_3D(
    __read_only image2d_array_t   inputA,
    __read_only image2d_array_t   inputB,
    __write_only image2d_array_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero
    )
{
    int4 coord_a = (int4)(0, get_global_id(1), (ac2zero ? 0 : get_global_id(2)), 0);
    int4 coord_b = (int4)(get_global_id(0), 0, (bc2zero ? 0 : get_global_id(2)), 0);

    float4 sum = (float4)(0);

    for(; coord_a.x < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord_a);
        tempB0 = read_imagef(inputB, coord_b);
        coord_a.x++;
        coord_b.y++;

        sum += tempA0 * tempB0;
    }

    coord_b.y = get_global_id(1);
    coord_b.z = get_global_id(2);
    write_imagef(output, coord_b, sum);
}
