inline float roi_align_1x1
(
    __read_only  image2d_array_t  input,
                           float2 region_start,
                           float2 region_end,
                           float2 bin_size,
                           int2   grid_size,
                           float2 rcp_of_grid_size,
                           int    pz
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

            float ly = pos.y - xy_low.y;
            float lx = pos.x - xy_low.x;
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;

            float w1 = hy * hx;
            float w2 = hy * lx;
            float w3 = ly * hx;
            float w4 = ly * lx;

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
__kernel void roi_align_F32toF32
(
    __read_only  image2d_array_t  input,
    __read_only  image2d_t        rois,
    __read_only  image2d_t        n_rois,
    __write_only image2d_array_t  output,
                           float  spatial_x_scale,
                           float  spatial_y_scale,
                           float  in_width,
                           float  in_height,
                           float  rcp_of_out_width,
                           float  rcp_of_out_height,
                           float  sampling_x_ratio,
                           float  sampling_y_ratio,
                           int    depth
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
    float2 max_spatial_dims = (float2)(in_width, in_height);

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
                       kz);

        write_imagef(output, (int4)(px, py, kz1, 0), interp);
    }
}