__kernel void slice_F32_I32toF32
    (
    __read_only  image2d_array_t input0,
    __read_only  image2d_t       input1,
    __write_only image2d_array_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in;
    Image begin_img = create_image_from_image2d(input1, 4);
    uchar* begin_ptr = begin_img.ptr;
    int4 begin = ((int4 *)begin_ptr)[0];

    coord_in = coord + begin;
    float4 src = read_imagef(input0, coord_in);

    write_imagef(output, coord, src);
}

__kernel void slice_F32_I32toF32_2D
    (
    __read_only  image2d_t input0,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in;
    Image begin_img = create_image_from_image2d(input1, 4);
    uchar* begin_ptr = begin_img.ptr;
    int2 begin = ((int2 *)begin_ptr)[0];

    coord_in = coord + begin;
    float4 src = read_imagef(input0, coord_in);

    write_imagef(output, coord, src);
}

__kernel void slice_U8_I32toU8
    (
    __read_only  image2d_array_t input0,
    __read_only  image2d_t       input1,
    __write_only image2d_array_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in;
    Image begin_img = create_image_from_image2d(input1, 4);
    uchar* begin_ptr = begin_img.ptr;
    int4 begin = ((int4 *)begin_ptr)[0];

    coord_in = coord + begin;
    uint4 src = read_imageui(input0, coord_in);

    float4 data = convert_float4(src) * inputScale - inputTail;
    uint4 dst = convert_uint4(data * outputScale + outputZP);

    write_imageui(output, coord, dst);
}

__kernel void slice_U8_I32toU8_2D
    (
    __read_only  image2d_t input0,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in;
    Image begin_img = create_image_from_image2d(input1, 4);
    uchar* begin_ptr = begin_img.ptr;
    int2 begin = ((int2 *)begin_ptr)[0];

    coord_in = coord + begin;
    uint4 src = read_imageui(input0, coord_in);

    float4 data = convert_float4(src) * inputScale - inputTail;
    uint4 dst = convert_uint4(data * outputScale + outputZP);

    write_imageui(output, coord, dst);
}

__kernel void slice_I32_I32toI32
    (
    __read_only  image2d_array_t input0,
    __read_only  image2d_t       input1,
    __write_only image2d_array_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 coord_in;
    Image begin_img = create_image_from_image2d(input1, 4);
    uchar* begin_ptr = begin_img.ptr;
    int4 begin = ((int4 *)begin_ptr)[0];

    coord_in = coord + begin;
    int4 src = read_imagei(input0, coord_in);

    write_imagei(output, coord, src);
}

__kernel void slice_I32_I32toI32_2D
    (
    __read_only  image2d_t input0,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 float           inputScale,
                 float           inputTail,
                 float           outputScale,
                 float           outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int2 coord_in;
    Image begin_img = create_image_from_image2d(input1, 4);
    uchar* begin_ptr = begin_img.ptr;
    int2 begin = ((int2 *)begin_ptr)[0];

    coord_in = coord + begin;
    int4 src = read_imagei(input0, coord_in);

    write_imagei(output, coord, src);
}

