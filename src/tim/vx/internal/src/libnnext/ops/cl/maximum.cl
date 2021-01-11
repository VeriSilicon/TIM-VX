__kernel void maximum_FP32FP32toFP32
    (
    __read_only  image2d_array_t    input0,
    __read_only  image2d_array_t    input1,
    __write_only image2d_array_t    output,
                 float              input0Scale,
                 float              input0Tail,
                 float              input1Scale,
                 float              input1Tail,
                 float              outputScale,
                 float              outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    float4 src0;
    float4 src1;
    readImage2DArray(src0, input0, coord);
    readImage2DArray(src1, input1, coord);

    float4 dst = src0 > src1 ? src0 : src1;

    write_imagef(output, coord, dst);
}

__kernel void maximum_FP32FP32toFP32_2D
    (
    __read_only  image2d_t    input0,
    __read_only  image2d_t    input1,
    __write_only image2d_t    output,
                 float        input0Scale,
                 float        input0Tail,
                 float        input1Scale,
                 float        input1Tail,
                 float        outputScale,
                 float        outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    float4 src0 = read_imagef(input0, coord);
    float4 src1 = read_imagef(input1, coord);

    float4 dst = src0 > src1 ? src0 : src1;

    write_imagef(output, coord, dst);
}

__kernel void maximum_U8U8toU8
    (
    __read_only  image2d_array_t    input0,
    __read_only  image2d_array_t    input1,
    __write_only image2d_array_t    output,
                 float              input0Scale,
                 float              input0Tail,
                 float              input1Scale,
                 float              input1Tail,
                 float              outputScale,
                 float              outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    uint4 src0;
    uint4 src1;
    readImage2DArray(src0, input0, coord);
    readImage2DArray(src1, input1, coord);

    float4 data0 = convert_float4(src0) * input0Scale - input0Tail;
    float4 data1 = convert_float4(src1) * input1Scale - input1Tail;
    float4 data = data0 > data1 ? data0 : data1;
    uint4 dst = convert_uint4(data * outputScale + outputZP);

    write_imageui(output, coord, dst);
}

__kernel void maximum_U8U8toU8_2D
    (
    __read_only  image2d_t    input0,
    __read_only  image2d_t    input1,
    __write_only image2d_t    output,
                 float        input0Scale,
                 float        input0Tail,
                 float        input1Scale,
                 float        input1Tail,
                 float        outputScale,
                 float        outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    uint4 src0 = read_imageui(input0, coord);
    uint4 src1 = read_imageui(input1, coord);

    float4 data0 = convert_float4(src0) * input0Scale - input0Tail;
    float4 data1 = convert_float4(src1) * input1Scale - input1Tail;
    float4 data = data0 > data1 ? data0 : data1;
    uint4 dst = convert_uint4(data * outputScale + outputZP);

    write_imageui(output, coord, dst);
}


__kernel void maximum_I32I32toI32
    (
    __read_only  image2d_array_t    input0,
    __read_only  image2d_array_t    input1,
    __write_only image2d_array_t    output,
                 float              input0Scale,
                 float              input0Tail,
                 float              input1Scale,
                 float              input1Tail,
                 float              outputScale,
                 float              outputZP
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    int4 src0;
    int4 src1;
    readImage2DArray(src0, input0, coord);
    readImage2DArray(src1, input1, coord);

    int4 dst = src0 > src1 ? src0 : src1;

    write_imagei(output, coord, dst);
}

__kernel void maximum_I32I32toI32_2D
    (
    __read_only  image2d_t    input0,
    __read_only  image2d_t    input1,
    __write_only image2d_t    output,
                 float        input0Scale,
                 float        input0Tail,
                 float        input1Scale,
                 float        input1Tail,
                 float        outputScale,
                 float        outputZP
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    int4 src0 = read_imagei(input0, coord);
    int4 src1 = read_imagei(input1, coord);

    int4 dst = src0 > src1 ? src0 : src1;

    write_imagei(output, coord, dst);
}

