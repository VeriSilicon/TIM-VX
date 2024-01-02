#pragma OPENCL EXTENSION CL_VIV_asm : enable

__kernel void gemm_4x_F32F32toF32_2D(
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
    int offset0 = get_global_id(0) * K;
    int offset1 = offset0 + K;
    int offset2 = offset1 + K;
    int offset3 = offset2 + K;
    int out_offset = get_global_id(0);
    int z = 0;
    float4 sum = (float4)(0, 0, 0, 0);

    Image in0_tensor = create_image_from_image2d(inputA, 4);
    __global float* in0_ptr = (__global float*)in0_tensor.ptr;
    __global float* in0_ptr0 = in0_ptr + offset0;
    __global float* in0_ptr1 = in0_ptr + offset1;
    __global float* in0_ptr2 = in0_ptr + offset2;
    __global float* in0_ptr3 = in0_ptr + offset3;

    Image in1_tensor = create_image_from_image2d(inputB, 4);
    __global float* in1_ptr = (__global float*)in1_tensor.ptr;

    Image o_tensor = create_image_from_image2d(output, 4);
    __global float* output_ptr = (__global float*)o_tensor.ptr + out_offset;

    int step = K >> 2;
    for(z = 0; z < step; z++)
    {
        float4 tempA0, tempA1, tempA2, tempA3;
        float4 tempB0;

        tempB0 = vload4(z, in1_ptr);
        tempA0 = vload4(z, in0_ptr0);
        tempA1 = vload4(z, in0_ptr1);
        tempA2 = vload4(z, in0_ptr2);
        tempA3 = vload4(z, in0_ptr3);

        sum.x += dot(tempA0, tempB0);
        sum.y += dot(tempA1, tempB0);
        sum.z += dot(tempA2, tempB0);
        sum.w += dot(tempA3, tempB0);
    }

    vstore4(sum, 0, output_ptr);

}

__kernel void gemm_4x_transa_F32F32toF32_2D(
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
    int offset0 = get_global_id(0);
    int offset1 = M << 2;

    int z = 0;
    float4 sum = (float4)(0, 0, 0, 0);

    Image in0_tensor = create_image_from_image2d(inputA, 4);
    __global float* in0_ptr0 = (__global float*)in0_tensor.ptr + offset0;
    __global float* in0_ptr1 = in0_ptr0 + M;
    __global float* in0_ptr2 = in0_ptr1 + M;
    __global float* in0_ptr3 = in0_ptr2 + M;

    Image in1_tensor = create_image_from_image2d(inputB, 4);
    __global float* in1_ptr = (__global float*)in1_tensor.ptr;

    Image o_tensor = create_image_from_image2d(output, 4);
    __global float* output_ptr = (__global float*)o_tensor.ptr + offset0;

    int step = K >> 2;
    for(z = 0; z < step; z++)
    {
        float4 tempA0, tempA1, tempA2, tempA3;
        float4 tempB0;

        tempB0 = vload4(z, in1_ptr);
        tempA0 = vload4(0, in0_ptr0);
        tempA1 = vload4(0, in0_ptr1);
        tempA2 = vload4(0, in0_ptr2);
        tempA3 = vload4(0, in0_ptr3);

        sum += tempA0 * tempB0.x;
        sum += tempA1 * tempB0.y;
        sum += tempA2 * tempB0.z;
        sum += tempA3 * tempB0.w;

        in0_ptr0 = in0_ptr0 + offset1;
        in0_ptr1 = in0_ptr1 + offset1;
        in0_ptr2 = in0_ptr2 + offset1;
        in0_ptr3 = in0_ptr3 + offset1;

    }

    vstore4(sum, 0, output_ptr);

}

__kernel __attribute__((reqd_work_group_size(1, 64, 1)))
    void gemm_4x_transa_local_F32F32toF32_2D(
    __read_only  image2d_t inputA,
    __read_only  image2d_t inputB,
    __write_only image2d_t output,
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
    int offset0 = get_global_id(0);
    int lid = get_local_id(1);

    int stride = 0;

    int z = 0;
    int offset1 = M << 2;
    int step = K >> 8;
    int lid2 = lid * 4 * step;

    Image in0_tensor = create_image_from_image2d(inputA, 4);
    __global float* in0_ptr0 = (__global float*)in0_tensor.ptr + offset0 + lid2 * M;
    __global float* in0_ptr1 = in0_ptr0 + M;
    __global float* in0_ptr2 = in0_ptr1 + M;
    __global float* in0_ptr3 = in0_ptr2 + M;

    Image in1_tensor = create_image_from_image2d(inputB, 4);
    __global float* in1_ptr = (__global float*)in1_tensor.ptr + lid2;

    Image o_tensor = create_image_from_image2d(output, 4);
    __global float* output_ptr = (__global float*)o_tensor.ptr + offset0;

    __local float4 sum_vec4_0[64];
    __local float4 sum_vec4_1[64];
    __local float4 sum_vec4_2[64];
    __local float4 sum_vec4_3[64];

    float4 sum0 = (float4)(0.0, 0.0, 0.0, 0.0);
    float4 sum1 = (float4)(0.0, 0.0, 0.0, 0.0);
    float4 sum2 = (float4)(0.0, 0.0, 0.0, 0.0);
    float4 sum3 = (float4)(0.0, 0.0, 0.0, 0.0);

    float4 tempA0, tempA1, tempA2, tempA3;
    float4 tempA4, tempA5, tempA6, tempA7;
    float4 tempB0;

    for(z = 0; z < step; z++)
    {
        tempB0 = vload4(z, in1_ptr);
        tempA0 = vload4(0, in0_ptr0);
        tempA1 = vload4(0, in0_ptr1);
        tempA2 = vload4(0, in0_ptr2);
        tempA3 = vload4(0, in0_ptr3);
        tempA4 = vload4(1, in0_ptr0);
        tempA5 = vload4(1, in0_ptr1);
        tempA6 = vload4(1, in0_ptr2);
        tempA7 = vload4(1, in0_ptr3);

        sum0 = sum0 + tempA0 * tempB0.x;
        sum0 = sum0 + tempA1 * tempB0.y;
        sum0 = sum0 + tempA2 * tempB0.z;
        sum0 = sum0 + tempA3 * tempB0.w;
        sum1 = sum1 + tempA4 * tempB0.x;
        sum1 = sum1 + tempA5 * tempB0.y;
        sum1 = sum1 + tempA6 * tempB0.z;
        sum1 = sum1 + tempA7 * tempB0.w;

        tempA0 = vload4(2, in0_ptr0);
        tempA1 = vload4(2, in0_ptr1);
        tempA2 = vload4(2, in0_ptr2);
        tempA3 = vload4(2, in0_ptr3);
        tempA4 = vload4(3, in0_ptr0);
        tempA5 = vload4(3, in0_ptr1);
        tempA6 = vload4(3, in0_ptr2);
        tempA7 = vload4(3, in0_ptr3);

        in0_ptr0 = in0_ptr0 + offset1;
        in0_ptr1 = in0_ptr1 + offset1;
        in0_ptr2 = in0_ptr2 + offset1;
        in0_ptr3 = in0_ptr3 + offset1;

        sum2 = sum2 + tempA0 * tempB0.x;
        sum2 = sum2 + tempA1 * tempB0.y;
        sum2 = sum2 + tempA2 * tempB0.z;
        sum2 = sum2 + tempA3 * tempB0.w;
        sum3 = sum3 + tempA4 * tempB0.x;
        sum3 = sum3 + tempA5 * tempB0.y;
        sum3 = sum3 + tempA6 * tempB0.z;
        sum3 = sum3 + tempA7 * tempB0.w;
    }
    sum_vec4_0[lid] = sum0;
    sum_vec4_1[lid] = sum1;
    sum_vec4_2[lid] = sum2;
    sum_vec4_3[lid] = sum3;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (stride = 32; stride > 0; stride >>= 1)
    {
        if (lid < stride)
        {
            sum_vec4_0[lid] += sum_vec4_0[lid + stride];
            sum_vec4_1[lid] += sum_vec4_1[lid + stride];
            sum_vec4_2[lid] += sum_vec4_2[lid + stride];
            sum_vec4_3[lid] += sum_vec4_3[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        sum0 = sum_vec4_0[0];
        sum1 = sum_vec4_1[0];
        sum2 = sum_vec4_2[0];
        sum3 = sum_vec4_3[0];
        vstore4(sum0, 0, output_ptr);
        vstore4(sum1, 1, output_ptr);
        vstore4(sum2, 2, output_ptr);
        vstore4(sum3, 3, output_ptr);
    }
}

