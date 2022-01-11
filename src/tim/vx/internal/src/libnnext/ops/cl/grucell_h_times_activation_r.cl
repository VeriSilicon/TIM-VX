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

#define GRUCELL_H_TIMES_R_U8_F32_F32(act_name, act_func) \
__kernel void grucell_h_times_activation_r_U8_F32toF32_##act_name( \
    __read_only  image2d_t        hstate_in, \
    __read_only  image2d_t        input_r_conv, \
    __read_only  image2d_t        hstate_r_conv, \
    __write_only image2d_t        output, \
    float input_scale, float input_tail) \
{ \
    int2 coord_in = (int2)(get_global_id(0), get_global_id(1)); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 r0 = read_imagef(input_r_conv, coord_in.xy); \
    float4 r1 = read_imagef(hstate_r_conv, coord_in.xy); \
    float4 h_tm = convert_float4(read_imageui(hstate_in, coord_in.xy)); \
 \
    float4 r = r0 + r1; \
    r.x = act_func(r.x); \
    h_tm = h_tm * input_scale + input_tail; \
    float4 r_times_h = r * h_tm; \
    write_imagef(output, coord_in.xy, r_times_h); \
}
GRUCELL_H_TIMES_R_U8_F32_F32(SIGMOID, sigmoid)
//GRUCELL_H_TIMES_R_U8_F32_F32(HARD_SIGMOID, hard_sigmoid)

#define GRUCELL_H_TIMES_R_F32_F32_F32(act_name, act_func) \
__kernel void grucell_h_times_activation_r_F32_F32toF32_##act_name( \
    __read_only  image2d_t hstate_in, \
    __read_only  image2d_t input_r_conv, \
    __read_only  image2d_t hstate_r_conv, \
    __write_only image2d_t output, \
    float input_scale, float input_tail) \
{ \
    int2 coord_in = (int2)(get_global_id(0), get_global_id(1)); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 r0 = read_imagef(input_r_conv, coord_in.xy); \
    float4 r1 = read_imagef(hstate_r_conv, coord_in.xy); \
    float4 h_tm = read_imagef(hstate_in, coord_in.xy); \
 \
    float4 r = r0 + r1; \
    r.x = act_func(r.x); \
    float4 r_times_h = r * h_tm; \
    write_imagef(output, coord_in.xy, r_times_h); \
}

GRUCELL_H_TIMES_R_F32_F32_F32(SIGMOID, sigmoid)
//GRUCELL_H_TIMES_R_F32_F32_F32(HARD_SIGMOID, hard_sigmoid)

#define GRUCELL_H_TIMES_R_I32_F32_F32(act_name, act_func) \
__kernel void grucell_h_times_activation_r_I32_F32toI32_##act_name( \
    __read_only  image2d_t        hstate_in, \
    __read_only  image2d_t        input_r_conv, \
    __read_only  image2d_t        hstate_r_conv, \
    __write_only image2d_t        output, \
    float input_scale, float input_tail) \
{ \
    int2 coord_in = (int2)(get_global_id(0), get_global_id(1)); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 r0 = read_imagef(input_r_conv, coord_in.xy); \
    float4 r1 = read_imagef(hstate_r_conv, coord_in.xy); \
    float4 h_tm = convert_float4(read_imagei(hstate_in, coord_in.xy)); \
 \
    float4 r = r0 + r1; \
    r.x = act_func(r.x); \
    h_tm = h_tm * input_scale + input_tail; \
    float4 r_times_h = r * h_tm; \
    write_imagef(output, coord_in.xy, r_times_h); \
}
GRUCELL_H_TIMES_R_I32_F32_F32(SIGMOID, sigmoid)
//GRUCELL_H_TIMES_R_I32_F32_F32(HARD_SIGMOID, hard_sigmoid)