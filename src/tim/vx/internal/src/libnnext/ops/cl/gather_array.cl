__kernel void gather_array_U8toU8(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);
    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    Image img1 = create_image_from_image2d(input0, 1);
    Image img2 = create_image_from_image2d(output, 1);
    __global uchar* input_ptr = get_image_ptr_from_coord(img1, coord_in.zw);
    uchar data = input_ptr[0];
    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    __global uchar* output_ptr = get_image_ptr_from_coord(img2, coord);
    output_ptr[0] = data;
}

__kernel void gather_array_F16toF16(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);
    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    Image img1 = create_image_from_image2d(input0, 2);
    Image img2 = create_image_from_image2d(output, 2);
    __global short* input_ptr = (__global short*)get_image_ptr_from_coord(img1, coord_in.zw);
    short data = input_ptr[0];
    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    __global short* output_ptr = (__global short*)get_image_ptr_from_coord(img2, coord);
    output_ptr[0] = data;
}

__kernel void gather_array_I32toI32(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);
    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    Image img1 = create_image_from_image2d(input0, 4);
    Image img2 = create_image_from_image2d(output, 4);
    __global int* input_ptr = (__global int*)get_image_ptr_from_coord(img1, coord_in.zw);
    int data = input_ptr[0];
    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    __global int* output_ptr = (__global int*)get_image_ptr_from_coord(img2, coord);
    output_ptr[0] = data;
}

__kernel void gather_array_F32toF32(
    __read_only image2d_t   input0,
    __read_only image2d_t   input1,
    __write_only image2d_t  output,
    int block_size,
    int block_num,
    int axis_num,
    int indices_num,
    int batch
    )
{
    int gidx = get_global_id(0);  // block_size
    int gidy = get_global_id(1);  // indices_num
    int gidz = get_global_id(2);  // block_num

    int4 coord_in = (int4)(gidy, 0, gidx, 0);
    int4 indice = read_imagei(input1, coord_in.xy);
    indice.x = indice.x >= 0 ? indice.x : indice.x + axis_num;
    coord_in.w = gidz * axis_num + indice.x;

    Image img1 = create_image_from_image2d(input0, 4);
    Image img2 = create_image_from_image2d(output, 4);
    __global float* input_ptr = (__global float*)get_image_ptr_from_coord(img1, coord_in.zw);
    float data = input_ptr[0];
    int2 coord = (int2)(gidx, gidz * indices_num + gidy);
    __global float* output_ptr = (__global float*)get_image_ptr_from_coord(img2, coord);
    output_ptr[0] = data;
}
