
__kernel void relu_keras_F32toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  alpha,
                           float  max_value,
                           float  threshold,
                           float  offset
                           )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = read_imagef(input, coord);
    float4 dst = src >= max_value ? max_value : src;
    dst = dst < threshold ? alpha * dst + offset : dst;
    write_imagef(output, coord, dst);
}

__kernel void relu_keras_F32toF32_2D(
    __read_only  image2d_t  input,
    __write_only image2d_t  output,
                     float  alpha,
                     float  max_value,
                     float  threshold,
                     float  offset
                     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = read_imagef(input, coord);
    float4 dst = src >= max_value ? max_value : src;
    dst = dst < threshold ? alpha * dst + offset : dst;
    write_imagef(output, coord, dst);
}

__kernel void relu_keras_F32toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  alpha,
                           float  max_value,
                           float  threshold,
                           float  offset,
                           float  inputScale,
                           float  inputTail,
                           float  outputScale,
                           float  outputZP
                           )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = read_imagef(input, coord);
    float4 result = src >= max_value ? max_value : src;
    result = result < threshold ? alpha * result + offset : result;
    uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
    write_imageui(output, coord, dst);
}

__kernel void relu_keras_F32toU8_2D(
    __read_only  image2d_t  input,
    __write_only image2d_t  output,
                     float  alpha,
                     float  max_value,
                     float  threshold,
                     float  offset,
                     float  inputScale,
                     float  inputTail,
                     float  outputScale,
                     float  outputZP
                     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = read_imagef(input, coord);
    float4 result = src >= max_value ? max_value : src;
    result = result < threshold ? alpha * result + offset : result;
    uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
    write_imageui(output, coord, dst);
}

__kernel void relu_keras_U8toU8(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  alpha,
                           float  max_value,
                           float  threshold,
                           float  offset,
                           float  inputScale,
                           float  inputTail,
                           float  outputScale,
                           float  outputZP
                           )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = convert_float4(read_imageui(input, coord))  * inputScale + inputTail;
    float4 result = src >= max_value ? max_value : src;
    result = result < threshold ? alpha * result + offset : result;
    uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
    write_imageui(output, coord, dst);
}

__kernel void relu_keras_U8toU8_2D(
    __read_only  image2d_t  input,
    __write_only image2d_t  output,
                     float  alpha,
                     float  max_value,
                     float  threshold,
                     float  offset,
                     float  inputScale,
                     float  inputTail,
                     float  outputScale,
                     float  outputZP
                     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = convert_float4(read_imageui(input, coord))  * inputScale + inputTail;
    float4 result = src >= max_value ? max_value : src;
    result = result < threshold ? alpha * result + offset : result;
    uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
    write_imageui(output, coord, dst);
}

__kernel void relu_keras_U8toF32(
    __read_only  image2d_array_t  input,
    __write_only image2d_array_t  output,
                           float  alpha,
                           float  max_value,
                           float  threshold,
                           float  offset,
                           float  inputScale,
                           float  inputTail,
                           float  outputScale,
                           float  outputZP
                           )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    float4 src = convert_float4(read_imageui(input, coord))  * inputScale + inputTail;
    float4 dst = src >= max_value ? max_value : src;
    dst = dst < threshold ? alpha * dst + offset : dst;
    write_imagef(output, coord, dst);
}

__kernel void relu_keras_U8toF32_2D(
    __read_only  image2d_t  input,
    __write_only image2d_t  output,
                     float  alpha,
                     float  max_value,
                     float  threshold,
                     float  offset,
                     float  inputScale,
                     float  inputTail,
                     float  outputScale,
                     float  outputZP
                     )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));
    float4 src = convert_float4(read_imageui(input, coord))  * inputScale + inputTail;
    float4 dst = src >= max_value ? max_value : src;
    dst = dst < threshold ? alpha * dst + offset : dst;
    write_imagef(output, coord, dst);
}