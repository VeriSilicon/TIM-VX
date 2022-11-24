__kernel void mod_F32F32toF32
    (
    __read_only  image2d_array_t  input,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 int              isfmod,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail,
                 float            outputScale,
                 float            outputTail
     )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src0;
    float4 src1;
    READ_IMAGEF_2DARRAY(src0, input, coord);
    READ_IMAGEF_2DARRAY(src1, input1, coord);
    float4 dst  = fmod(src0, src1);
    write_imagef(output, coord, dst);
}

__kernel void mod_F32F32toF32_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 int       isfmod,
                 float     input0Scale,
                 float     input0Tail,
                 float     input1Scale,
                 float     input1Tail,
                 float     outputScale,
                 float     outputTail
     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src0 = read_imagef(input, coord);
    float4 src1 = read_imagef(input1, coord);
    float4 dst  = fmod(src0, src1);
    write_imagef(output, coord, dst);
}

__kernel void mod_I32I32toI32
    (
    __read_only  image2d_array_t  input,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 int              isfmod,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail,
                 float            outputScale,
                 float            outputTail
     )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 src0;
    int4 src1;
    READ_IMAGEI_2DARRAY(src0, input, coord);
    READ_IMAGEI_2DARRAY(src1, input1, coord);
    float4 in0 = convert_float4(src0) * input0Scale + input0Tail;
    float4 in1 = convert_float4(src1) * input1Scale + input1Tail;
    float4 out;
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    int4 dst = convert_int4(out);
    write_imagei(output, coord, dst);
}

__kernel void mod_I32I32toI32_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 int       isfmod,
                 float     input0Scale,
                 float     input0Tail,
                 float     input1Scale,
                 float     input1Tail,
                 float     outputScale,
                 float     outputTail
     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int4 src0 = read_imagei(input, coord);
    int4 src1 = read_imagei(input1, coord);
    float4 in0 = convert_float4(src0) * input0Scale + input0Tail;
    float4 in1 = convert_float4(src1) * input1Scale + input1Tail;
    float4 out;
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    int4 dst = convert_int4(out);
    write_imagei(output, coord, dst);
}

__kernel void mod_I32I32toU8
    (
    __read_only  image2d_array_t  input,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 int              isfmod,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail,
                 float            outputScale,
                 float            outputTail
     )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 src0;
    int4 src1;
    READ_IMAGEI_2DARRAY(src0, input, coord);
    READ_IMAGEI_2DARRAY(src1, input1, coord);
    float4 in0 = convert_float4(src0) * input0Scale + input0Tail;
    float4 in1 = convert_float4(src1) * input1Scale + input1Tail;
    float4 out;
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    uint4 dst = convert_uint4(out);
    write_imageui(output, coord, dst);
}

__kernel void mod_I32I32toU8_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 int       isfmod,
                 float     input0Scale,
                 float     input0Tail,
                 float     input1Scale,
                 float     input1Tail,
                 float     outputScale,
                 float     outputTail
     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    int4 src0 = read_imagei(input, coord);
    int4 src1 = read_imagei(input1, coord);
    float4 in0 = convert_float4(src0) * input0Scale + input0Tail;
    float4 in1 = convert_float4(src1) * input1Scale + input1Tail;
    float4 out;
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    uint4 dst = convert_uint4(out);
    write_imageui(output, coord, dst);
}

__kernel void mod_U8U8toU8
    (
    __read_only  image2d_array_t  input,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 int              isfmod,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail,
                 float            outputScale,
                 float            outputTail
     )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    uint4 src0, src1;
    float4 in0, in1, out;
    READ_IMAGEUI_2DARRAY(src0, input, coord);
    READ_IMAGEUI_2DARRAY(src1, input1, coord);
    in0 = convert_float4(src0) * input0Scale + input0Tail;
    in1 = convert_float4(src1) * input1Scale + input1Tail;
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    uint4 dst  = convert_uint4(out);
    write_imageui(output, coord, dst);
}

__kernel void mod_U8U8toU8_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 int       isfmod,
                 float     input0Scale,
                 float     input0Tail,
                 float     input1Scale,
                 float     input1Tail,
                 float     outputScale,
                 float     outputTail
     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    uint4 src0 = read_imageui(input, coord);
    uint4 src1 = read_imageui(input1, coord);
    float4 in0, in1, out;
    in0 = convert_float4(src0) * input0Scale + input0Tail;
    in1 = convert_float4(src1) * input1Scale + input1Tail;
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    uint4 dst  = convert_uint4(out);
    write_imageui(output, coord, dst);
}

__kernel void mod_U8I32toU8
    (
    __read_only  image2d_array_t  input,
    __read_only  image2d_array_t  input1,
    __write_only image2d_array_t  output,
                 int              isfmod,
                 float            input0Scale,
                 float            input0Tail,
                 float            input1Scale,
                 float            input1Tail,
                 float            outputScale,
                 float            outputTail
     )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    uint4 src0;
    int4 src1;
    float4 in0, in1, out;
    READ_IMAGEUI_2DARRAY(src0, input, coord);
    READ_IMAGEI_2DARRAY(src1, input1, coord);
    in0 = convert_float4(src0) * input0Scale + input0Tail;
    in1 = convert_float4(src1);
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    uint4 dst = convert_uint4(out);
    write_imageui(output, coord, dst);
}

__kernel void mod_U8I32toU8_2D
    (
    __read_only  image2d_t input,
    __read_only  image2d_t input1,
    __write_only image2d_t output,
                 int       isfmod,
                 float     input0Scale,
                 float     input0Tail,
                 float     input1Scale,
                 float     input1Tail,
                 float     outputScale,
                 float     outputTail
     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    uint4 src0 = read_imageui(input, coord);
    int4 src1 = read_imagei(input1, coord);
    float4 in0, in1, out;
    in0 = convert_float4(src0) * input0Scale + input0Tail;
    in1 = convert_float4(src1);
    if (isfmod)
    {
        out = fmod(in0, in1) * outputScale + outputTail;
    }
    else
    {
        out = (in0 - in1 * floor(in0 / in1)) * outputScale + outputTail;
    }
    uint4 dst = convert_uint4(out);
    write_imageui(output, coord, dst);
}
