/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include <float.h>
#include "utils/vsi_nn_dtype_util_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_lut.h"

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel,
    vsi_enum                      lut_type
    )
{
#ifdef VX_USER_LOOKUP_TABLE_SUPPORT
    vx_lut lut1 = NULL;
    vx_lut lut2 = NULL;
    vx_node node = NULL;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_lut_params lut_param;

    memset(&lut_param, 0, sizeof(lut_param));

    lut_param.act_type = lut_type;
    if (lut_type == VSI_NN_KERNEL_LUT_RELU_KERAS)
    {
        lut_param.params[0] = vsi_nn_kernel_param_get_float32( params, "alpha" );
        lut_param.params[1] = vsi_nn_kernel_param_get_float32( params, "max_value" );
        lut_param.params[2] = vsi_nn_kernel_param_get_float32( params, "threshold" );
    }
    else if (lut_type == VSI_NN_KERNEL_LUT_CLIP)
    {
        lut_param.params[0] = vsi_nn_kernel_param_get_float32( params, "min_value" );
        lut_param.params[1] = vsi_nn_kernel_param_get_float32( params, "max_value" );
    }
    else if (lut_type == VSI_NN_KERNEL_LUT_SELU || lut_type == VSI_NN_KERNEL_LUT_HSIGMOID)
    {
        lut_param.params[0] = vsi_nn_kernel_param_get_float32( params, "alpha" );
        lut_param.params[1] = vsi_nn_kernel_param_get_float32( params, "beta" );
    }
    else if (lut_type == VSI_NN_KERNEL_LUT_CELU)
    {
        lut_param.params[0] = vsi_nn_kernel_param_get_float32( params, "alpha" );
    }
    else if (lut_type == VSI_NN_KERNEL_LUT_ACOSH)
    {
        lut_param.pwl_sign_remove_support = TRUE;
        lut_param.clamp_min = 0;
    }

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32   ||
         outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32  )
    {
        return NULL;
    }

    lut1 = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_KERNEL_LUT_MAX_SIZE);
    lut2 = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_KERNEL_LUT_MAX_SIZE);
    if( NULL == lut1 || NULL == lut2 )
    {
        VSILOGE("create lut object fail.");
        goto final;
    }

    status = vsi_nn_kernel_lut(lut1, lut2, &lut_param);
    CHECK_STATUS_FAIL_GOTO(status, final);

    node = vxTensorTableLookupLayer( graph->g, inputs[0]->t, lut1, lut2, outputs[0]->t);
    if ( NULL == node )
    {
        VSILOGI("Call vxTensorTableLookupLayer fail.");
        goto final;
    }

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

final:
    if (lut1)
    {
        vxReleaseLUT(&lut1);
        lut1 = NULL;
    }
    if (lut2)
    {
        vxReleaseLUT(&lut2);
        lut2 = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
#else
    VSI_UNREFERENCED(graph);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(outputs);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(lut_type);
    return NULL;
#endif
} /* _setup() */

#define REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL(KERNEL_NAME, UNARY_FUNC) \
    static vsi_nn_kernel_node_t _##KERNEL_NAME##_setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num, \
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ) \
    { \
        return _setup(graph, inputs, input_num, outputs, output_num, \
                params, kernel, UNARY_FUNC); \
    } \
    REGISTER_BACKEND_OPENVX( KERNEL_NAME, _##KERNEL_NAME##_setup )

REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( mish,         VSI_NN_KERNEL_LUT_MISH )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( log,          VSI_NN_KERNEL_LUT_LOG )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( selu,         VSI_NN_KERNEL_LUT_SELU )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( neg,          VSI_NN_KERNEL_LUT_NEG )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( hard_sigmoid, VSI_NN_KERNEL_LUT_HSIGMOID )
#if !(VX_ACTIVATION_GELU_VX_SUPPORT_EXT)
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( gelu,         VSI_NN_KERNEL_LUT_GELU )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( hard_gelu,    VSI_NN_KERNEL_LUT_HGELU )
#endif
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( erf,          VSI_NN_KERNEL_LUT_ERF )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( relu_keras,   VSI_NN_KERNEL_LUT_RELU_KERAS )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( clip,         VSI_NN_KERNEL_LUT_CLIP )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( celu,         VSI_NN_KERNEL_LUT_CELU )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( rcp,          VSI_NN_KERNEL_LUT_RCP )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( softsign,     VSI_NN_KERNEL_LUT_SOFTSIGN )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( atan,         VSI_NN_KERNEL_LUT_ATAN )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( tan,          VSI_NN_KERNEL_LUT_TAN )

#undef REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL

#define REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( kernel_name )   \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ); \
    REGISTER_BACKEND_OPENVX( kernel_name, _##kernel_name##setup ) \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        )

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( abs )
{
    vx_node node = NULL;
    vsi_size_t input_size[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dims = 0;
    vx_tensor input = NULL, input0 = NULL;
    vx_tensor output = NULL, output0 = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    if (inputs[0]->attr.dim_num > 4)
    {
        input_size[0] = vsi_nn_GetElementNum(inputs[0]) /
            inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        input_size[1] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        dims = 2;
#ifdef VSI_40BIT_VA_SUPPORT
        input = vxReshapeTensor(inputs[0]->t, input_size, dims);
        output = vxReshapeTensor(outputs[0]->t, input_size, dims);
#else
        input = vxReshapeTensor(inputs[0]->t, (vx_int32*)input_size, (vx_uint32)dims);
        output = vxReshapeTensor(outputs[0]->t, (vx_int32*)input_size, (vx_uint32)dims);
#endif
        input0 = input;
        output0 = output;
    }
    else
    {
        input0 = inputs[0]->t;
        output0 = outputs[0]->t;
    }

    node = vxLeakyReluLayer(
        graph->g,
        input0,
        -1,
        output0
        );

    if (input)  vxReleaseTensor(&input);
    if (output) vxReleaseTensor(&output);

    return (vsi_nn_kernel_node_t)node;
} /* abs() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( linear )
{
    vx_node node = NULL;
    float a_v = vsi_nn_kernel_param_get_float32( params, "a_v" );
    float b_v = vsi_nn_kernel_param_get_float32( params, "b_v" );

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LINEAR,
        a_v,
        b_v,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* linear() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( sigmoid )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LOGISTIC,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* sigmoid() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( tanh )
{
    vx_node node = NULL;
    float scale_a = vsi_nn_kernel_param_get_float32( params, "scale_a" );
    float scale_b = vsi_nn_kernel_param_get_float32( params, "scale_b" );

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HYPERBOLIC_TAN,
        scale_a,
        scale_b,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* tanh() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( relu1 )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RELU1,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* relu1() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( relu6 )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RELU6,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* relu6() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( rsqrt )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RSQRT,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* rsqrt() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( sqrt )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SQRT,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* sqrt() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( softrelu )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SOFTRELU,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* softrelu() */

#if (VX_ACTIVATION_EXP_VX_SUPPORT_EXT)
REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( exp )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_EXP,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* exp() */
#endif

#if (VX_ACTIVATION_SIN_COS_VX_SUPPORT_EXT)
REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( sin )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SIN,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* sin() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( cos )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_COS,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* cos() */
#endif

#if (VX_ACTIVATION_GELU_VX_SUPPORT_EXT)
REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( gelu )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_GELU,
        1,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* gelu() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( hard_gelu )
{
    vx_node node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HGELU,
        1,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* hard_gelu() */
#endif

#undef REGISTER_ELTWISE_UNARY_OPENVX_KERNEL
