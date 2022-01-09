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
    else if (lut_type == VSI_NN_KERNEL_LUT_ELU || lut_type == VSI_NN_KERNEL_LUT_HSIGMOID)
    {
        lut_param.params[0] = vsi_nn_kernel_param_get_float32( params, "alpha" );
        lut_param.params[1] = vsi_nn_kernel_param_get_float32( params, "beta" );
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
        VSILOGW("Call vxTensorTableLookupLayer fail.");
        goto final;
    }

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
//REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( exp,          VSI_NN_KERNEL_LUT_EXP )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( log,          VSI_NN_KERNEL_LUT_LOG )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( elu,          VSI_NN_KERNEL_LUT_ELU )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( neg,          VSI_NN_KERNEL_LUT_NEG )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( hard_sigmoid, VSI_NN_KERNEL_LUT_HSIGMOID )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( gelu,         VSI_NN_KERNEL_LUT_GELU )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( hard_gelu,    VSI_NN_KERNEL_LUT_HGELU )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( erf,          VSI_NN_KERNEL_LUT_ERF )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( relu_keras,   VSI_NN_KERNEL_LUT_RELU_KERAS )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( clip,         VSI_NN_KERNEL_LUT_CLIP )

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

#undef REGISTER_ELTWISE_UNARY_OPENVX_KERNEL
