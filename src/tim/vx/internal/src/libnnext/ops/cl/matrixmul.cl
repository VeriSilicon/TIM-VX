__kernel void gemm_F32F32toF32_2D(
    __read_only image2d_t   inputA,
    __read_only image2d_t   inputB,
    __write_only image2d_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    float4 sum = (float4)(0);

    for(; coord.z < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord.zy);
        tempB0 = read_imagef(inputB, coord.xz);
        coord.z++;

        sum = sum + tempA0 * tempB0;
    }
    write_imagef(output, coord.xy, sum);
}

__kernel void gemm_F32F32toF32_3D(
    __read_only image2d_array_t   inputA,
    __read_only image2d_array_t   inputB,
    __write_only image2d_array_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
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

        sum = sum + tempA0 * tempB0;
    }

    coord_b.y = get_global_id(1);
    coord_b.z = get_global_id(2);
    write_imagef(output, coord_b, sum);
}

__kernel void gemm_transb_F32F32toF32_2D(
    __read_only image2d_t   inputA,
    __read_only image2d_t   inputB,
    __write_only image2d_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    float4 sum = (float4)(0);

    for(; coord.z < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord.zy);
        tempB0 = read_imagef(inputB, coord.zx);
        coord.z++;

        sum = sum + tempA0 * tempB0;
    }
    write_imagef(output, coord.xy, sum);
}

__kernel void gemm_transb_F32F32toF32_3D(
    __read_only image2d_array_t   inputA,
    __read_only image2d_array_t   inputB,
    __write_only image2d_array_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord_a = (int4)(0, get_global_id(1), (ac2zero ? 0 : get_global_id(2)), 0);
    int4 coord_b = (int4)(0, get_global_id(0), (bc2zero ? 0 : get_global_id(2)), 0);

    float4 sum = (float4)(0);

    for(; coord_a.x < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord_a);
        tempB0 = read_imagef(inputB, coord_b);
        coord_a.x++;
        coord_b.x++;

        sum = sum + tempA0 * tempB0;
    }

    coord_a.x = get_global_id(0);
    coord_a.z = get_global_id(2);
    write_imagef(output, coord_b, sum);
}

__kernel void gemm_transb_F32I8toF32_2D(
    __read_only image2d_t   inputA,
    __read_only image2d_t   inputB,
    __write_only image2d_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    float4 sum = (float4)(0);
    for(; coord.z < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord.zy);
        tempB0 = convert_float4(read_imagei(inputB, coord.zx));
        coord.z++;
        tempB0.x = (tempB0.x - zp_b) * scale_b;

        sum = sum + tempA0 * tempB0;
    }

    write_imagef(output, coord.xy, sum);
}

__kernel void gemm_transb_F32I8toF32_3D(
    __read_only image2d_array_t   inputA,
    __read_only image2d_array_t   inputB,
    __write_only image2d_array_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord_a = (int4)(0, get_global_id(1), (ac2zero ? 0 : get_global_id(2)), 0);
    int4 coord_b = (int4)(0, get_global_id(0), (bc2zero ? 0 : get_global_id(2)), 0);

    float4 sum = (float4)(0);

    for(; coord_a.x < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord_a);
        tempB0 = convert_float4(read_imagei(inputB, coord_b));
        tempB0.x = (tempB0.x - zp_b) * scale_b;
        coord_a.x++;
        coord_b.x++;

        sum = sum + tempA0 * tempB0;
    }

    coord_a.x = get_global_id(0);
    coord_a.z = get_global_id(2);
    write_imagef(output, coord_b, sum);
}
