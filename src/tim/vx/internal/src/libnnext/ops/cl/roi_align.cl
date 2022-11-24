
inline float roi_align_1x1
(
    __read_only  image2d_array_t  input,
                 float2 region_start,
                 float2 region_end,
                 float2 bin_size,
                 int2   grid_size,
                 float2 rcp_of_grid_size,
                 int    pz,
                 int4   max_spatial_dims
)
{
    float sum = 0;

    for(int iy = 0; iy < grid_size.y; ++iy)
    {
        for(int ix = 0; ix < grid_size.x; ++ix)
        {
            float2 ixy = (float2)(ix + 0.5f, iy + 0.5f);
            float2 pos = region_start + ixy * bin_size * rcp_of_grid_size;

            int2 xy_low  = convert_int2(pos);
            int2 xy_high = xy_low + 1;

            if (xy_low.x > max_spatial_dims.x || max_spatial_dims.x < -1 ||
                xy_low.y > max_spatial_dims.y || max_spatial_dims.y < -1 )
            {
                continue;
            }

            float2 lxy = pos - floor(pos);
            float2 zero = 0;

            lxy = xy_low >= max_spatial_dims.zw ? 0.0 : lxy;

            float hy = 1.0f - lxy.y;
            float hx = 1.0f - lxy.x;

            float w1 = hy * hx;
            float w2 = lxy.x - lxy.x * lxy.y;
            float w3 = lxy.y - lxy.x * lxy.y;
            float w4 = lxy.y * lxy.x;

            float data1 = read_imagef(input, (int4)(xy_low.x, xy_low.y, pz, 0)).x;
            float data2 = read_imagef(input, (int4)(xy_high.x, xy_low.y, pz, 0)).x;
            float data3 = read_imagef(input, (int4)(xy_low.x, xy_high.y, pz, 0)).x;
            float data4 = read_imagef(input, (int4)(xy_high.x, xy_high.y, pz, 0)).x;

            sum = sum + w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
        }
    }

    return (float)(sum * rcp_of_grid_size.x * rcp_of_grid_size.y);
}

#define EPS_GRID 0.00001f
#define TYPE_FLOAT16    (1)
#define TYPE_FLOAT32    (2)
__kernel void roi_align_F32_F32toF32
(
    __read_only  image2d_array_t input,
    __read_only  image2d_t       rois,
    __read_only  image2d_t       n_rois,
    __write_only image2d_array_t output,
                 float           input_scale,
                 float           input_tail,
                 float           output_scale,
                 float           output_zp,
                 float           spatial_x_scale,
                 float           spatial_y_scale,
                 int             in_width,
                 int             in_height,
                 float           rcp_of_out_width,
                 float           rcp_of_out_height,
                 float           sampling_x_ratio,
                 float           sampling_y_ratio,
                 int             depth,
                 int             dtype
)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    int pw = get_global_id(2);

    int roi_batch = read_imagei(n_rois, (int2)(pw, 0)).x;
    float4 roi_x = read_imagef(rois, (int2)(0, pw));
    float4 roi_y = read_imagef(rois, (int2)(1, pw));
    float4 roi_z = read_imagef(rois, (int2)(2, pw));
    float4 roi_w = read_imagef(rois, (int2)(3, pw));
    float4 roi = (float4)(roi_x.x, roi_y.x, roi_z.x, roi_w.x);

    float4 roi_anchor = roi * (float4)(spatial_x_scale, spatial_y_scale, spatial_x_scale, spatial_y_scale);
    float2 roi_dims = fmax(roi_anchor.zw - roi_anchor.xy, 1.0f);

    float2 spatial_indx     = (float2)(px, py);
    float2 pooled_dims      = (float2)(rcp_of_out_width, rcp_of_out_height);
    int4 max_spatial_dims   = (int4)(in_width, in_height, in_width, in_height);
    max_spatial_dims.zw = max_spatial_dims.zw - 1;

    float2 max_limiatation = convert_float2(max_spatial_dims.zw);

    float2 bin_size     = roi_dims * pooled_dims;
    float2 region_start = spatial_indx * bin_size + roi_anchor.xy;
    float2 region_end   = region_start + bin_size;

    float2 roi_bin_grid = (float2)(sampling_x_ratio, sampling_y_ratio);

    roi_bin_grid = roi_bin_grid == 0 ? ceil(bin_size - EPS_GRID) : roi_bin_grid;

    int kz = roi_batch * depth;
    float2 rcp_of_grid_size = 1.0f / roi_bin_grid;
    int2 grid_size_xy = convert_int2(roi_bin_grid);
    float4 interp;
    int kz1 = pw * depth;
    for (int pz = 0; pz < depth; pz ++, kz ++, kz1 ++)
    {
        interp.x = roi_align_1x1( input,
                       region_start,
                       region_end,
                       bin_size,
                       grid_size_xy,
                       rcp_of_grid_size,
                       kz,
                       max_spatial_dims);

        if (dtype == TYPE_FLOAT16)
        {
            half tmp;
            short dst;
            _viv_asm(CONV, tmp, interp.x);
            _viv_asm(COPY, dst, tmp, 2);

            Tensor out_t =  create_tensor_from_image2d_array(output, 2);
            short *output_ptr = (short *)get_tensor_ptr_from_coord(out_t, (int4)(px, py, kz1, 0));

            output_ptr[0] = dst;
        }
        else
        {
            Tensor out_t =  create_tensor_from_image2d_array(output, 4);
            float *output_ptr = (float *)get_tensor_ptr_from_coord(out_t, (int4)(px, py, kz1, 0));

            output_ptr[0] = interp.x;
        }
    }
}

inline float roi_align_1x1_U8toF32
(
    __read_only image2d_array_t  input,
                float            input_scale,
                float            input_tail,
                float2           region_start,
                float2           region_end,
                float2           bin_size,
                int2             grid_size,
                float2           rcp_of_grid_size,
                int              pz,
                int4             max_spatial_dims
)
{
    float sum = 0;

    for(int iy = 0; iy < grid_size.y; ++iy)
    {
        for(int ix = 0; ix < grid_size.x; ++ix)
        {
            float2 ixy = (float2)(ix + 0.5f, iy + 0.5f);
            float2 pos = region_start + ixy * bin_size * rcp_of_grid_size;
    
            int2 xy_low  = convert_int2(pos);
            int2 xy_high = xy_low + 1;
    
            float2 lxy = pos - floor(pos);
            float2 zero = 0;
    
            if (xy_low.x > max_spatial_dims.x || max_spatial_dims.x < -1 ||
                xy_low.y > max_spatial_dims.y || max_spatial_dims.y < -1 )
            {
                continue;
            }
    
            lxy = xy_low >= max_spatial_dims.zw ? 0.0 : lxy;
    
            float hy = 1.0f - lxy.y;
            float hx = 1.0f - lxy.x;
    
            float w1 = hy * hx;
            float w2 = lxy.x - lxy.x * lxy.y;
            float w3 = lxy.y - lxy.x * lxy.y;
            float w4 = lxy.y * lxy.x;
    
            uint4 data;
            data.x = read_imageui(input, (int4)(xy_low.x, xy_low.y, pz, 0)).x;
            data.y = read_imageui(input, (int4)(xy_high.x, xy_low.y, pz, 0)).x;
            data.z = read_imageui(input, (int4)(xy_low.x, xy_high.y, pz, 0)).x;
            data.w = read_imageui(input, (int4)(xy_high.x, xy_high.y, pz, 0)).x;
    
            float4 value = convert_float4(data) * input_scale + input_tail;
    
            sum = sum + w1 * value.x + w2 * value.y + w3 * value.z + w4 * value.w;
        }
    }
    
    return (float)(sum * rcp_of_grid_size.x * rcp_of_grid_size.y);

}

__kernel void roi_align_U8_U16toU8
(
    __read_only  image2d_array_t input,
    __read_only  image2d_t       rois,
    __read_only  image2d_t       n_rois,
    __write_only image2d_array_t output,
                 float           input_scale,
                 float           input_tail,
                 float           output_scale,
                 float           output_zp,
                 float           spatial_x_scale,
                 float           spatial_y_scale,
                 int             in_width,
                 int             in_height,
                 float           rcp_of_out_width,
                 float           rcp_of_out_height,
                 float           sampling_x_ratio,
                 float           sampling_y_ratio,
                 int             depth,
                 int             dtype
)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    int pw = get_global_id(2);

    int roi_batch = read_imagei(n_rois, (int2)(pw, 0)).x;
    float4 roi_x = convert_float4(read_imageui(rois, (int2)(0, pw)));
    float4 roi_y = convert_float4(read_imageui(rois, (int2)(1, pw)));
    float4 roi_z = convert_float4(read_imageui(rois, (int2)(2, pw)));
    float4 roi_w = convert_float4(read_imageui(rois, (int2)(3, pw)));
    float4 roi = (float4)(roi_x.x, roi_y.x, roi_z.x, roi_w.x);

    float4 roi_anchor = roi * (float4)(spatial_x_scale, spatial_y_scale, spatial_x_scale, spatial_y_scale);
    float2 roi_dims = fmax(roi_anchor.zw - roi_anchor.xy, 1.0f);

    float2 spatial_indx     = (float2)(px, py);
    float2 pooled_dims      = (float2)(rcp_of_out_width, rcp_of_out_height);
    int4 max_spatial_dims   = (int4)(in_width, in_height, in_width, in_height);
    max_spatial_dims.zw = max_spatial_dims.zw - 1;

    float2 max_limiatation = convert_float2(max_spatial_dims.zw);

    float2 bin_size     = roi_dims * pooled_dims;
    float2 region_start = spatial_indx * bin_size + roi_anchor.xy;
    float2 region_end   = region_start + bin_size;

    float2 roi_bin_grid = (float2)(sampling_x_ratio, sampling_y_ratio);

    roi_bin_grid = roi_bin_grid == 0 ? ceil(bin_size - EPS_GRID) : roi_bin_grid;

    int kz = roi_batch * depth;
    float2 rcp_of_grid_size = 1.0f / roi_bin_grid;
    int2 grid_size_xy = convert_int2(roi_bin_grid);
    float4 interp;
    int kz1 = pw * depth;
    for (int pz = 0; pz < depth; pz ++, kz ++, kz1 ++)
    {
        interp.x = roi_align_1x1_U8toF32( input,
                       input_scale,
                       input_tail,
                       region_start,
                       region_end,
                       bin_size,
                       grid_size_xy,
                       rcp_of_grid_size,
                       kz,
                       max_spatial_dims);

        uchar dst;
        interp.x = interp.x * output_scale + output_zp;
        interp.x = interp.x < 255 ? interp.x : 255;
        dst = convert_uchar_rte(interp.x);

        Tensor out_t =  create_tensor_from_image2d_array(output, 1);
        uchar *output_ptr = (uchar *)get_tensor_ptr_from_coord(out_t, (int4)(px, py, kz1, 0));
        
        output_ptr[0] = dst;
    }
}