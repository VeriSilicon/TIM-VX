__kernel void group_norm_sumsqr_U8(
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
    float tmpSum = 0, tmpSqr = 0;
    float e2InScale = input_scale * input_scale;

    __local float lcl_sum[16];
    __local float lcl_sqr[16];

    if(gidx < width)
    {
        for(coord.y = 0; coord.y < height;)
        {
            data = convert_float4(read_imageui(input, coord));
            coord.y++;
            tmpSum += data.x;
            tmpSqr += data.x * data.x;
        }
        sqr = (tmpSqr - 2 * input_zp * tmpSum + height * input_zp * input_zp) * e2InScale;
        sum = (tmpSum - height * input_zp) * input_scale;
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

__kernel void group_norm_sumsqr_U8_2D(
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
        data = convert_float4(read_imageui(input, coord));
        sum = (data.x - input_zp) * input_scale;
        sqr = sum * sum;
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

__kernel void group_norm_U8toU8(
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
    float4 data = convert_float4(read_imageui(input, coord));

    float scale_vari, bias_val;
    float scale_inOut = input_scale * output_scale;

    scale_vari = gamma.s0 * mean_vari.s1;
    float alpha = scale_inOut * scale_vari;
    bias_val = (beta.s0 - scale_vari * mean_vari.s0) * output_scale + output_zp;

    uint4 dst;
    data.x -= input_zp;
    float4 norm;
    norm.x = data.x * alpha + bias_val;
    dst = convert_uint4_rte(norm);
    write_imageui(output, coord, dst);
}

__kernel void group_norm_U8toU8_2D(
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
    float4 data = convert_float4(read_imageui(input, coord));

    float scale_vari, bias_val;
    float scale_inOut = input_scale * output_scale;

    scale_vari = gamma.s0 * mean_vari.s1;
    float alpha = scale_inOut * scale_vari;
    bias_val = (beta.s0 - scale_vari * mean_vari.s0) * output_scale + output_zp;

    uint4 dst;
    data.x -= input_zp;
    float4 norm;
    norm.x = data.x * alpha + bias_val;
    dst = convert_uint4_rte(norm);
    write_imageui(output, coord, dst);
}

__kernel void group_norm_U8toF32(
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
    float4 data = convert_float4(read_imageui(input, coord));

    float scale_vari, bias_val;
    float scale_inOut = input_scale * output_scale;

    scale_vari = gamma.s0 * mean_vari.s1;
    float alpha = scale_inOut * scale_vari;
    bias_val = (beta.s0 - scale_vari * mean_vari.s0) * output_scale + output_zp;

    data.x -= input_zp;
    float4 norm;
    norm.x = data.x * alpha + bias_val;
    write_imagef(output, coord, norm);
}

__kernel void group_norm_U8toF32_2D(
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
    float4 data = convert_float4(read_imageui(input, coord));

    float scale_vari, bias_val;
    float scale_inOut = input_scale * output_scale;

    scale_vari = gamma.s0 * mean_vari.s1;
    float alpha = scale_inOut * scale_vari;
    bias_val = (beta.s0 - scale_vari * mean_vari.s0) * output_scale + output_zp;

    data.x -= input_zp;
    float4 norm;
    norm.x = data.x * alpha + bias_val;
    write_imagef(output, coord, norm);
}
