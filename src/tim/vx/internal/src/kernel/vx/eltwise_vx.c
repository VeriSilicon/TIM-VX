/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include "kernel/vsi_nn_kernel.h"


#define REGISTER_ELTWISE_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_ELTWISE_OPENVX_KERNEL( add )
{
    vx_node node = vxTensorAddNode( graph->g, inputs[0]->t, inputs[1]->t,
        VX_CONVERT_POLICY_SATURATE, outputs[0]->t );
    return (vsi_nn_kernel_node_t)node;
} /* add() */

REGISTER_ELTWISE_OPENVX_KERNEL( sub )
{
    vx_node node = vxTensorSubtractNode( graph->g, inputs[0]->t, inputs[1]->t,
                VX_CONVERT_POLICY_SATURATE, outputs[0]->t );

    return (vsi_nn_kernel_node_t)node;
} /* sub() */

REGISTER_ELTWISE_OPENVX_KERNEL( div )
{
    float scale;
    vsi_enum overflow_policy, rounding_policy;
    vx_scalar scale_s = NULL;
    vx_node node = NULL;

    scale = vsi_nn_kernel_param_get_float32(params, "scale");
    overflow_policy = vsi_nn_kernel_param_get_int32(params, "overflow_policy");
    rounding_policy = vsi_nn_kernel_param_get_int32(params, "rounding_policy");

    scale_s = vxCreateScalar(graph->ctx->c, VX_TYPE_FLOAT32, &scale);
    if(!scale_s)
    {
        VSILOGE("CreateScalar fail\n");
        return NULL;
    }

    node = vxTensorDivideNode( graph->g,
        inputs[0]->t, inputs[1]->t,
        scale_s,
        overflow_policy,
        rounding_policy,
        outputs[0]->t );

    vxReleaseScalar(&scale_s);

    return (vsi_nn_kernel_node_t)node;
} /* div() */

REGISTER_ELTWISE_OPENVX_KERNEL( mul )
{
    float scale;
    vsi_enum overflow_policy, rounding_policy;
    vx_scalar scale_s = NULL;
    vx_node node = NULL;

    scale = vsi_nn_kernel_param_get_float32(params, "scale");
    overflow_policy = vsi_nn_kernel_param_get_int32(params, "overflow_policy");
    rounding_policy = vsi_nn_kernel_param_get_int32(params, "rounding_policy");

    scale_s = vxCreateScalar(graph->ctx->c, VX_TYPE_FLOAT32, &scale);
    if(!scale_s)
    {
        VSILOGE("CreateScalar fail\n");
        return NULL;
    }

    node = vxTensorMultiplyNode( graph->g,
        inputs[0]->t, inputs[1]->t,
        scale_s,
        overflow_policy,
        rounding_policy,
        outputs[0]->t );

    vxReleaseScalar(&scale_s);

    return (vsi_nn_kernel_node_t)node;
} /* mul() */

#undef REGISTER_ELTWISE_OPENVX_KERNEL

