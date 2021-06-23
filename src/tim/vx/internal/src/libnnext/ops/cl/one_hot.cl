__kernel void one_hot_F32toF32
    (
        __read_only  image2d_t       input,
        __write_only image2d_array_t output,
                     int             depth,
                     float           on_value,
                     float           off_value,
                     float           inputScale,
                     float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    float4 val = read_imagef(input, coord.xy);

    do
    {
        float4 dst;
        dst.x = convert_int(val.x) == coord.z ? on_value : off_value;

        write_imagef(output, coord.xzyw, dst.xxxx);

        coord.z ++;
    } while (coord.z < depth);
}

__kernel void one_hot_I32toI32
    (
        __read_only  image2d_t       input,
        __write_only image2d_array_t output,
                     int             depth,
                     int             on_value,
                     int             off_value,
                     float           inputScale,
                     float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    int4 val = read_imagei(input, coord.xy);

    do
    {
        int4 dst;
        dst.x = val.x == coord.z ? on_value : off_value;

        write_imagei(output, coord.xzyw, dst.xxxx);

        coord.z ++;
    } while (coord.z < depth);
}

__kernel void one_hot_I32toU8
    (
        __read_only  image2d_t       input,
        __write_only image2d_array_t output,
                     int             depth,
                     uint            on_value,
                     uint            off_value,
                     float           inputScale,
                     float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    int4 val = read_imagei(input, coord.xy);
    do
    {
        uint4 dst;
        dst.x = val.x == coord.z ? on_value : off_value;

        write_imageui(output, coord.xzyw, dst.xxxx);

        coord.z ++;
    } while (coord.z < depth);
}

__kernel void one_hot_I32toF32
    (
        __read_only  image2d_t       input,
        __write_only image2d_array_t output,
                     int             depth,
                     float           on_value,
                     float           off_value,
                     float           inputScale,
                     float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    int4 val = read_imagei(input, coord.xy);

    do
    {
        float4 dst;
        dst.x = val.x == coord.z ? on_value : off_value;

        write_imagef(output, coord.xzyw, dst.xxxx);

        coord.z ++;
    } while (coord.z < depth);
}

__kernel void one_hot_U8toU8
    (
        __read_only  image2d_t       input,
        __write_only image2d_array_t output,
                     int             depth,
                     uint            on_value,
                     uint            off_value,
                     float           inputScale,
                     float           inputTail
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), 0, 0);

    uint4 src = read_imageui(input, coord.xy);

    int  val = convert_int(convert_float(src.x) * inputScale - inputTail);

    do
    {
        uint4 dst;
        dst.x = val == coord.z ? on_value : off_value;

        write_imageui(output, coord.xzyw, dst.xxxx);

        coord.z ++;
    } while (coord.z < depth);
}
