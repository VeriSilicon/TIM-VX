__kernel void select_I8_U8_U8toU8(
    __read_only  image2d_array_t  condition,
    __read_only  image2d_array_t  input0,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail)
{
    int4 coord  = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4  value;
    uint4 src0, src1, src, dst;
    float inputScale, inputTail;
    READ_IMAGEI_2DARRAY(value, condition, coord);
    READ_IMAGEF_2DARRAY(src0, input0, coord);
    READ_IMAGEF_2DARRAY(src1, input1, coord);
    src   = (value != 0 ? src0 : src1);
    inputScale = (value.x != 0 ? input0Scale : input1Scale);
    inputTail  = (value.x != 0 ? input0Tail  : input1Tail);
    dst = convert_uint4(convert_float4(src) * inputScale + inputTail);
    write_imageui(output, coord, dst);
}

__kernel void select_I8_U8_U8toU8_2D(
    __read_only  image2d_t        condition,
    __read_only  image2d_t        input0,
    __read_only  image2d_t        input1,
    __write_only image2d_t        output,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int4  value = read_imagei(condition, coord);
    uint4 src0  = read_imageui(input0, coord);
    uint4 src1  = read_imageui(input1, coord);
    uint4 src   = (value != 0 ? src0 : src1);
    float inputScale = (value.x != 0 ? input0Scale : input1Scale);
    float inputTail  = (value.x != 0 ? input0Tail  : input1Tail);
    uint4 dst = convert_uint4(convert_float4(src) * inputScale + inputTail);
    write_imageui(output, coord, dst);
}

__kernel void select_I8_I32_I32toI32(
    __read_only  image2d_array_t  condition,
    __read_only  image2d_array_t  input0,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail)
{
    int4 coord  = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4  value;
    int4 src0, src1, dst;
    READ_IMAGEI_2DARRAY(value, condition, coord);
    READ_IMAGEI_2DARRAY(src0, input0, coord);
    READ_IMAGEI_2DARRAY(src1, input1, coord);
    dst   = (value != 0 ? src0 : src1);
    write_imagei(output, coord, dst);
}

__kernel void select_I8_I32_I32toI32_2D(
    __read_only  image2d_t        condition,
    __read_only  image2d_t        input0,
    __read_only  image2d_t        input1,
    __write_only image2d_t        output,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int4 value = read_imagei(condition, coord);
    int4 src0  = read_imagei(input0, coord);
    int4 src1  = read_imagei(input1, coord);
    int4 dst   = (value != 0 ? src0 : src1);
    write_imagei(output, coord, dst);
}

__kernel void select_I8_F32_F32toF32(
    __read_only  image2d_array_t  condition,
    __read_only  image2d_array_t  input0,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail)
{
    int4 coord  = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4  value;
    float4 src0, src1, dst;
    READ_IMAGEI_2DARRAY(value, condition, coord);
    READ_IMAGEF_2DARRAY(src0, input0, coord);
    READ_IMAGEF_2DARRAY(src1, input1, coord);
    dst   = (value != 0 ? src0 : src1);
    write_imagef(output, coord, dst);
}

__kernel void select_I8_F32_F32toF32_2D(
    __read_only  image2d_t        condition,
    __read_only  image2d_t        input0,
    __read_only  image2d_t        input1,
    __write_only image2d_t        output,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail)
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int4 value = read_imagei(condition, coord);
    float4 src0  = read_imagef(input0, coord);
    float4 src1  = read_imagef(input1, coord);
    float4 dst   = (value != 0 ? src0 : src1);
    write_imagef(output, coord, dst);
}
