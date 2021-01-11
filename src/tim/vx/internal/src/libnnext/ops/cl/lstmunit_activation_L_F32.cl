float4 sigmoid(float4 x, float logE)
{
    x *= -logE;
    x = 1 + exp2(x);
    return 1 / x;
}
float4 hard_sigmoid(float4 x, float logE)
{
    x = 0.2 * x + 0.5;
    x = clamp(x, 0, 1);
    return x;
}
float4 tangentH(float4 x, float twoLogE)
{
    x *= -twoLogE;
    x = 1 + exp2(x);
    x = 1 / x;
    return 2 * x - 1;
}


#define LSTM_ACTIVATION_L_F32(act_name, act_func) \
__kernel void lstmunit_activation_L_F32toF32_F32_##act_name( \
    __read_only  image2d_t        input_i_conv, \
    __read_only  image2d_t        input_f_conv, \
    __read_only  image2d_t        input_c_conv, \
    __read_only  image2d_t        input_o_conv, \
    __read_only  image2d_t        cell_state_in, \
    __read_only  image2d_t        bias_i, \
    __read_only  image2d_t        bias_f, \
    __read_only  image2d_t        bias_c, \
    __read_only  image2d_t        bias_o, \
    __read_only  image2d_t        layer_norm_wi, \
    __read_only  image2d_t        layer_norm_wf, \
    __read_only  image2d_t        layer_norm_wc, \
    __read_only  image2d_t        layer_norm_wo, \
    __write_only image2d_t        output, \
    __write_only image2d_t        cell_state_out, \
    __write_only image2d_t        h_state_out, \
    float logE, float twoLogE, float forget_bias, float clip_Max_F, float clip_Min_F) \
{ \
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \
    float4  src0, src1, src2, src3; \
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \
    float4 b0, b1, b2, b3; \
    float4 w0, w1, w2, w3; \
    src0  = read_imagef(input_i_conv,  coord_in.xy); \
    src1  = read_imagef(input_f_conv,  coord_in.xy); \
    src2  = read_imagef(input_c_conv,  coord_in.xy); \
    src3  = read_imagef(input_o_conv,  coord_in.xy); \
    w0 = read_imagef(layer_norm_wi, coord_in.xw); \
    w1 = read_imagef(layer_norm_wf, coord_in.xw); \
    w2 = read_imagef(layer_norm_wc, coord_in.xw); \
    w3 = read_imagef(layer_norm_wo, coord_in.xw); \
    b0 = read_imagef(bias_i, coord_in.xw); \
    b1 = read_imagef(bias_f, coord_in.xw); \
    b2 = read_imagef(bias_c, coord_in.xw); \
    b3 = read_imagef(bias_o, coord_in.xw); \
    data_c_t = read_imagef(cell_state_in, coord_in.xy); \
    data_i_t = src0 * w0 + b0; \
    data_f_t = src1 * w1 + b1; \
    data_g_t = src2 * w2 + b2; \
    data_o_t = src3 * w3 + b3; \
    data_i_t = act_func(data_i_t, logE); \
    data_f_t = act_func(data_f_t + forget_bias, logE); \
    data_g_t = tangentH(data_g_t, twoLogE); \
    data_i_t = data_i_t * data_g_t; \
    data_c_t = data_c_t * data_f_t + data_i_t; \
    data_o_t = act_func(data_o_t, logE); \
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \
    data_c_t = tangentH(data_c_t, twoLogE); \
    data_o_t = data_o_t * data_c_t; \
    write_imagef(output, coord_in.zy, data_o_t); \
    write_imagef(h_state_out, coord_in.zy, data_o_t); \
}

LSTM_ACTIVATION_L_F32(SIGMOID, sigmoid)
LSTM_ACTIVATION_L_F32(HARD_SIGMOID, hard_sigmoid)
