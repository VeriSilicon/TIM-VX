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



