#pragma OPENCL EXTENSION CL_VIV_asm : enable

inline uchar* get_image2D_array_ptr(image2d_array_t  input)
{
    int8 desc;
    _viv_asm(COPY, desc, input, sizeof(desc));
    uchar *src_ptr = (uchar*)desc.s0;

    return src_ptr;
}

uint4 _philox4x32bumpkey(uint4 key)
{
    uint4 mask = (uint4)((uint)0x9E3779B9, (uint)0xBB67AE85, 0, 0);
    //key.x += ((uint)0x9E3779B9);
    //key.y += ((uint)0xBB67AE85);
    key += mask;
    return key;
}

uint mullo32(uint a, uint b)
{
    return a * b;
}

uint mulhi32(uint a, uint b)
{
    return mul_hi(a, b);
}

uint4 _philox4x32round(uint4 ctr, uint4 key)
{
    uint PHILOX_M4x32_0 = ((uint)0xD2511F53);
    uint PHILOX_M4x32_1 = ((uint)0xCD9E8D57);

    uint lo0 = mullo32(PHILOX_M4x32_0, ctr.x);
    uint hi0 = mulhi32(PHILOX_M4x32_0, ctr.x);
    uint lo1 = mullo32(PHILOX_M4x32_1, ctr.z);
    uint hi1 = mulhi32(PHILOX_M4x32_1, ctr.z);

    uint4 out = (uint4)(hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0);
    return out;
}

uint4 philox4x32_R_10(uint4 ctr, uint4 key)
{
    uint i;
    ctr = _philox4x32round(ctr, key);
    for (i = 1; i < 10; i++)
    {
        key = _philox4x32bumpkey(key);
        ctr = _philox4x32round(ctr, key);
    }
    return ctr;
}

__kernel void random_seed(
    __read_only  image2d_array_t  seeds,
    __write_only image2d_array_t  output,
                 int              iter,
                 float            re_rand_max
    )
{
    __global uint* seeds_ptr = (__global uint*)get_image2D_array_ptr(seeds);
    seeds_ptr = seeds_ptr;
    uint4 key = vload4(0, seeds_ptr);

    uint4 ctr = (uint4)(0);
    float4 result = 0;

    __global float* output_ptr = (__global float*)get_image2D_array_ptr(output);

    for(int i = 0; i < iter; i++)
    {
        ctr = philox4x32_R_10(ctr, key);
        result = convert_float4(ctr) * re_rand_max;
        vstore4(result, i, output_ptr);
    }
}

#define logE    (1.44269502f)
float eltwise_unary_exp(float x)
{
    x *= logE;
    x = exp2(x);
    return x;
}
// N times of 8
// x dim = 1

__kernel void random_multinomial_cdf_F32
    (
    __read_only  image2d_t  input,
    __write_only image2d_t  output
    )
{
    int2 coord = (int2)(0, get_global_id(1));
    int class_max_iter = get_image_width(input);
    float4 src0, data;
    float4 dst = 0;

    float4 maxVal = read_imagef(input, coord);

    for(coord.x = 1; coord.x < class_max_iter;)
    {
        src0 = read_imagef(input, coord);
        coord.x ++;

        maxVal = maxVal > src0 ? maxVal : src0;
    }

    for(coord.x = 0; coord.x < class_max_iter; )
    {
        float4 val;
        src0 = read_imagef(input, coord);

        data = src0 - maxVal;
        val.x = eltwise_unary_exp(data.x);
        val.x += dst.x;
        dst.x = val.x;
        write_imagef(output, coord.xy, val);
        coord.x ++;
    }
}

uint upper_bound(float* a, int n, float x)
{
    uint l = 0;
    uint h = n;
    while (l < h) {
        int mid = (l + h) >> 1;
        if (x >= a[mid]) {
            l = mid + 1;
        } else {
            h = mid;
        }
    }
    return l;
}

// one thread calculate 4
__kernel void random_multinomial
    (
    __read_only image2d_array_t randoms,
    __read_only image2d_array_t cdfs,
   __write_only image2d_array_t output
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int4 coord = (int4)(gidx, gidy, 0, 0);
    int class_size = get_image_width(cdfs);

    int offset = gidy * class_size;
    __global float* cdf_ptr = (__global float*)get_image2D_array_ptr(cdfs);
    __global float* cdfPtr = cdf_ptr + offset;

    int width = get_image_width(randoms);
    offset = coord.x + coord.y * width;
    __global float* randoms_ptr = (__global float*)get_image2D_array_ptr(randoms);
    randoms_ptr = randoms_ptr + offset;

    width = get_image_width(output);
    offset = coord.x + coord.y * width;
    __global uint* output_ptr = (__global uint*)get_image2D_array_ptr(output);
    output_ptr = output_ptr + offset;

    float4 ran = vload4(0, randoms_ptr);
    float total = cdfPtr[class_size - 1];
    float4 target = ran * total;

    uint4 out_class = (uint4)(0);
    out_class.x = upper_bound(cdfPtr, class_size, target.x);
    out_class.y = upper_bound(cdfPtr, class_size, target.y);
    out_class.z = upper_bound(cdfPtr, class_size, target.z);
    out_class.w = upper_bound(cdfPtr, class_size, target.w);

    vstore4(out_class, 0, output_ptr);
}

