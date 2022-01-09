#define logE        (1.44269502f)
#define twoLogE     (logE * 2.0f)

float sigmoid(float x)
{
    x *= -logE;
    x = 1 + exp2(x);
    return 1 / x;
}
float hard_sigmoid(float x)
{
    x = 0.2 * x + 0.5;
    x = clamp(x, 0, 1);
    return x;
}
float tanh_func(float x)
{
    x *= -twoLogE;
    x = 1 + exp2(x);
    x = 1 / x;
    return 2 * x - 1;
}


#define GRUCELL_ACTIVATION_U8_F32_U8(act_name, act_func) \
__kernel void grucell_activation_z_h_U8_F32toU8_##act_name( \
    __read_only  image2d_t        hstate_in, \
    __read_only  image2d_t        input_z_conv, \
    __read_only  image2d_t        input_h_conv, \
    __read_only  image2d_t        hstate_z_conv, \
    __read_only  image2d_t        hstate_h_conv, \
    __write_only image2d_t        output, \
    __write_only image2d_t        hstate_out, \
    float input_scale, float input_tail, float output_scale, float output_zp) \
{ \
    int2 coord_in = (int2)(get_global_id(0), get_global_id(1)); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 h_tm = convert_float4(read_imageui(hstate_in, coord_in.xy)); \
    float4 h1 = read_imagef(hstate_h_conv, coord_in.xy); \
    float4 h0 = read_imagef(input_h_conv, coord_in.xy); \
    float4 z0 = read_imagef(input_z_conv, coord_in.xy); \
    float4 z1 = read_imagef(hstate_z_conv, coord_in.xy); \
 \
    h_tm = h_tm * input_scale + input_tail; \
    float4 h = h0 + h1; \
    float4 z = z0 + z1; \
    z.x = act_func(z.x); \
    h = tanh_func(h.x); \
    float4 dst = (1 - z ) * h + z * h_tm; \
    dst = dst * output_scale + output_zp; \
    uint4 result = convert_uint4_sat_rte(dst); \
    write_imageui(output, coord_in.xy, result); \
    write_imageui(hstate_out, coord_in.xy, result); \
}
GRUCELL_ACTIVATION_U8_F32_U8(SIGMOID, sigmoid)
//GRUCELL_ACTIVATION_U8_F32_U8(HARD_SIGMOID, hard_sigmoid)

#define GRUCELL_ACTIVATION_F32_F32_F32(act_name, act_func) \
__kernel void grucell_activation_z_h_F32_F32toF32_##act_name( \
    __read_only  image2d_t        hstate_in, \
    __read_only  image2d_t        input_z_conv, \
    __read_only  image2d_t        input_h_conv, \
    __read_only  image2d_t        hstate_z_conv, \
    __read_only  image2d_t        hstate_h_conv, \
    __write_only image2d_t        output, \
    __write_only image2d_t        hstate_out, \
    float input_scale, float input_tail, float output_scale, float output_zp) \
{ \
    int2 coord_in = (int2)(get_global_id(0), get_global_id(1)); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 h1 = read_imagef(hstate_h_conv, coord_in.xy); \
    float4 h0 = read_imagef(input_h_conv, coord_in.xy); \
    float4 z0 = read_imagef(input_z_conv, coord_in.xy); \
    float4 z1 = read_imagef(hstate_z_conv, coord_in.xy); \
    float4 h_tm = read_imagef(hstate_in, coord_in.xy); \
 \
    float4 h = h0 + h1; \
    float4 z = z0 + z1; \
    z.x = act_func(z.x); \
    h = tanh_func(h.x); \
    float4 dst = (1 - z ) * h + z * h_tm; \
    write_imagef(output, coord_in.xy, dst); \
    write_imagef(hstate_out, coord_in.xy, dst); \
}

GRUCELL_ACTIVATION_F32_F32_F32(SIGMOID, sigmoid)
//GRUCELL_ACTIVATION_U8_F32_U8(HARD_SIGMOID, hard_sigmoid)

#define GRUCELL_ACTIVATION_I32_F32_I32(act_name, act_func) \
__kernel void grucell_activation_z_h_I32_F32toI32_##act_name( \
    __read_only  image2d_t        hstate_in, \
    __read_only  image2d_t        input_z_conv, \
    __read_only  image2d_t        input_h_conv, \
    __read_only  image2d_t        hstate_z_conv, \
    __read_only  image2d_t        hstate_h_conv, \
    __write_only image2d_t        output, \
    __write_only image2d_t        hstate_out, \
    float input_scale, float input_tail, float output_scale, float output_zp) \
{ \
    int2 coord_in = (int2)(get_global_id(0), get_global_id(1)); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 h_tm = convert_float4(read_imagei(hstate_in, coord_in.xy)); \
    float4 h1 = read_imagef(hstate_h_conv, coord_in.xy); \
    float4 h0 = read_imagef(input_h_conv, coord_in.xy); \
    float4 z0 = read_imagef(input_z_conv, coord_in.xy); \
    float4 z1 = read_imagef(hstate_z_conv, coord_in.xy); \
 \
    h_tm = h_tm * input_scale + input_tail; \
    float4 h = h0 + h1; \
    float4 z = z0 + z1; \
    z.x = act_func(z.x); \
    h = tanh_func(h.x); \
    float4 dst = (1 - z ) * h + z * h_tm; \
    dst = dst * output_scale + output_zp; \
    int4 result = convert_int4_sat_rte(dst); \
    write_imagei(output, coord_in.xy, result); \
    write_imagei(hstate_out, coord_in.xy, result); \
}
GRUCELL_ACTIVATION_I32_F32_I32(SIGMOID, sigmoid)
//GRUCELL_ACTIVATION_U8_F32_U8(HARD_SIGMOID, hard_sigmoid)