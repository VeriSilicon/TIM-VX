__kernel void gemm_transa_F32F32toF32_2D(
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

        tempA0 = read_imagef(inputA, coord.yz);
        tempB0 = read_imagef(inputB, coord.xz);
        coord.z++;

        sum = sum + tempA0 * tempB0;
    }
    write_imagef(output, coord.xy, sum);
}

__kernel void gemm_transa_F32F32toF32_3D(
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
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int4 coord_a = (int4)(gidy, 0, (ac2zero ? 0 : get_global_id(2)), 0);
    int4 coord_b = (int4)(gidx, 0, (bc2zero ? 0 : get_global_id(2)), 0);

    float4 sum = (float4)(0);

    for(; coord_a.y < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord_a);
        tempB0 = read_imagef(inputB, coord_b);
        coord_a.y++;
        coord_b.y++;

        sum = sum + tempA0 * tempB0;
    }

    coord_b.y = gidy;
    coord_b.z = get_global_id(2);
    write_imagef(output, coord_b, sum);
}

#define GEMM_TRANSA_2D(name, dst_type, read_image_type, convert_type, write_image_type) \
__kernel void gemm_transa_##name##_2D( \
    __read_only image2d_t   inputA, \
    __read_only image2d_t   inputB, \
    __write_only image2d_t  output, \
    int M, \
    int K, \
    int N, \
    int ac2zero, \
    int bc2zero, \
    float scale_a, \
    float zp_a, \
    float scale_b, \
    float zp_b, \
    float scale_out, \
    float zp_out \
    ) \
{ \
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \
    float4 sum = (float4)(0); \
    dst_type dst; \
 \
    for(; coord.z < K;) \
    { \
        float4 tempA0; \
        float4 tempB0; \
 \
        tempA0 = convert_float4(read_image_type(inputA, coord.yz)); \
        tempB0 = convert_float4(read_image_type(inputB, coord.xz)); \
        tempA0.x = (tempA0.x - zp_a) * scale_a; \
        tempB0.x = (tempB0.x - zp_b) * scale_b; \
        coord.z++; \
 \
        sum = sum + tempA0 * tempB0; \
    } \
    sum.x = sum.x * scale_out + zp_out; \
    dst = convert_type(sum); \
 \
    write_image_type(output, coord.xy, dst); \
}
GEMM_TRANSA_2D(I8I8toI8,int4,read_imagei,convert_int4,write_imagei);
GEMM_TRANSA_2D(U8U8toU8,uint4,read_imageui,convert_uint4,write_imageui);
GEMM_TRANSA_2D(U8U8toF32,float4,read_imageui,convert_float4,write_imagef);

#define GEMM_TRANSA_3D(name, dst_type, read_image_type, convert_type, write_image_type) \
__kernel void gemm_transa_##name##_3D( \
    __read_only image2d_array_t   inputA, \
    __read_only image2d_array_t   inputB, \
    __write_only image2d_array_t  output, \
    int M, \
    int K, \
    int N, \
    int ac2zero, \
    int bc2zero, \
    float scale_a, \
    float zp_a, \
    float scale_b, \
    float zp_b, \
    float scale_out, \
    float zp_out \
    ) \
{ \
    int gidx = get_global_id(0); \
    int gidy = get_global_id(1); \
 \
    int4 coord_a = (int4)(gidy, 0, (ac2zero ? 0 : get_global_id(2)), 0); \
    int4 coord_b = (int4)(gidx, 0, (bc2zero ? 0 : get_global_id(2)), 0); \
 \
    float4 sum = (float4)(0); \
    dst_type dst; \
 \
    for(; coord_a.y < K;) \
    { \
        float4 tempA0; \
        float4 tempB0; \
 \
        tempA0 = convert_float4(read_image_type(inputA, coord_a)); \
        tempB0 = convert_float4(read_image_type(inputB, coord_b)); \
        tempA0.x = (tempA0.x - zp_a) * scale_a; \
        tempB0.x = (tempB0.x - zp_b) * scale_b; \
        coord_a.y++; \
        coord_b.y++; \
 \
        sum = sum + tempA0 * tempB0; \
    } \
    sum.x = sum.x * scale_out + zp_out; \
    dst = convert_type(sum); \
 \
    coord_b.y = gidy; \
    coord_b.z = get_global_id(2); \
    write_image_type(output, coord_b, dst); \
}
GEMM_TRANSA_3D(I8I8toI8,int4,read_imagei,convert_int4,write_imagei);
GEMM_TRANSA_3D(U8U8toU8,uint4,read_imageui,convert_uint4,write_imageui);
GEMM_TRANSA_3D(U8U8toF32,float4,read_imageui,convert_float4,write_imagef);
