__kernel void group_norm_sumsqr_F32(
    __read_only image2d_array_t   input,
    __write_only image2d_t  output,
    float eps,
    int is2d,
    float input_zp,
    float input_scale,
    int width,
    int height
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);

    int4 coord = (int4)(gidx, 0, gidz, 0);
    float4 data;
    float sum = 0, sqr = 0;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    if(gidx < width)
    {
        for(coord.y = 0; coord.y < height;)
        {
            data = read_imagef(input, coord);
            coord.y++;
            sum += data.x;
            sqr += data.x * data.x;
        }
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int4 coord_out = (int4)(get_group_id(0) << 2, gidz, 0, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 dst = (float4)(0);
        dst.x = sum;
        write_imagef(output, coord_out.xy, dst);
        coord_out.x++;
        dst.x = sqr;
        write_imagef(output, coord_out.xy, dst);
    }
}

__kernel void group_norm_sumsqr_F32_2D(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    float eps,
    int is2d,
    float input_zp,
    float input_scale,
    int width,
    int height
    )
{
    int gidx = get_global_id(0);
    int gidz = get_global_id(1);
    int lidx = get_local_id(0);

    int2 coord = (int2)(gidx, gidz);
    float4 data;
    float sum = 0, sqr = 0;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    if(gidx < width)
    {
        data = read_imagef(input, coord);
        sum = data.x;
        sqr = data.x * data.x;
    }
    lcl_sum[lidx] = sum;
    lcl_sqr[lidx] = sqr;
    barrier(CLK_LOCAL_MEM_FENCE);

    int4 coord_out = (int4)(get_group_id(0) << 2, gidz, 0, 0);
    if(lidx == 0)
    {
        float4 one = (float4)(1, 1, 1, 1);
        __local float4* tmp_sum = (__local float4*)lcl_sum;
        __local float4* tmp_sqr = (__local float4*)lcl_sqr;

        sum = 0; sqr = 0;
        for(int i = 0; i < 4; i++)
        {
            sum += dot(tmp_sum[i], one);
            sqr += dot(tmp_sqr[i], one);
        }

        float4 dst = (float4)(0);
        dst.x = sum;
        write_imagef(output, coord_out.xy, dst);
        coord_out.x++;
        dst.x = sqr;
        write_imagef(output, coord_out.xy, dst);
    }
}

__kernel void group_norm_meanvari(
    __read_only image2d_t   input,
    __write_only image2d_t  output,
    float eps,
    float group_ratio,
    int group_stride
    )
{
    int gidx = get_global_id(0);
    int lidx = get_local_id(0);
    int2 coord = (int2)(gidx, get_global_id(1));

    float2 sum_sqr = (float2)(0);
    float4 mean_vari = (float4)(0);

    __local float2 lcl_data[16];
    __local float2 lcl_sum[4];

    for(; coord.x < group_stride;)
    {
        mean_vari.x += read_imagef(input, coord).x;
        coord.x++;
        mean_vari.y += read_imagef(input, coord).x;
        coord.x+=63;
    }
    lcl_data[lidx] = mean_vari.xy;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx < 4)
    {
        float2 tmpSum = (float2)(0);
        for(int i = lidx; i < 16; i+=4)
        {
            tmpSum += lcl_data[i];
        }
        lcl_sum[lidx] = tmpSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx == 0)
    {
        for(int i = 0; i < 4; i++)
        {
            sum_sqr += lcl_sum[i];
        }
        mean_vari.xy = sum_sqr * group_ratio;
        mean_vari.s1 = mean_vari.s1 - mean_vari.s0 * mean_vari.s0 + eps;
        mean_vari.s1 = rsqrt(mean_vari.s1);

        coord.x = 0;
        write_imagef(output, coord, mean_vari);
        coord.x++;
        float4 data;
        data.x = mean_vari.y;
        write_imagef(output, coord, data);
    }
}

__kernel void group_norm_F32toF32(
    __read_only image2d_array_t   input,
    __read_only image2d_t   bias,
    __read_only image2d_t   scale,
    __read_only image2d_t   meanVari,
    __write_only image2d_array_t  output,
    float eps,
    int is2d,
    float input_zp,
    float input_scale,
    float output_zp,
    float output_scale,
    float rSpaceOrg,
    int width,
    int height,
    int pStride
    )
{
    int gidy = get_global_id(1);
    int gidz = get_global_id(2);
    int4 coord = (int4)(get_global_id(0), gidy, gidz, 0);
    int4 coord_para = (int4)((convert_int(get_global_id(0) * rSpaceOrg) + gidy * pStride), gidz, 0, 1);

    float4 gamma = read_imagef(scale, coord_para.xy);
    float4 beta  = read_imagef(bias, coord_para.xy);
    float4 mean_vari = read_imagef(meanVari, coord_para.zy);
    mean_vari.y = read_imagef(meanVari, coord_para.wy).x;
    float4 data = read_imagef(input, coord);

    float scale_vari, bias_val;

    scale_vari = gamma.s0 * mean_vari.s1;
    bias_val = (beta.s0 - scale_vari * mean_vari.s0);

    float4 dst;

    dst.x = data.x * scale_vari + bias_val;
    write_imagef(output, coord, dst);
}

__kernel void group_norm_F32toF32_2D(
    __read_only image2d_t   input,
    __read_only image2d_t   bias,
    __read_only image2d_t   scale,
    __read_only image2d_t   meanVari,
    __write_only image2d_t  output,
    float eps,
    int is2d,
    float input_zp,
    float input_scale,
    float output_zp,
    float output_scale,
    float rSpaceOrg,
    int width,
    int height,
    int pStride
    )
{
    int gidz = get_global_id(1);
    int2 coord = (int2)(get_global_id(0), gidz);
    int4 coord_para = (int4)(convert_int(get_global_id(0) * rSpaceOrg), gidz, 0, 1);

    float4 gamma = read_imagef(scale, coord_para.xy);
    float4 beta  = read_imagef(bias, coord_para.xy);
    float4 mean_vari = read_imagef(meanVari, coord_para.zy);
    mean_vari.y = read_imagef(meanVari, coord_para.wy).x;
    float4 data = read_imagef(input, coord);

    float scale_vari, bias_val;

    scale_vari = gamma.s0 * mean_vari.s1;
    bias_val = beta.s0 - scale_vari * mean_vari.s0;

    float4 dst;

    dst.x = data.x * scale_vari + bias_val;
    write_imagef(output, coord, dst);
}
