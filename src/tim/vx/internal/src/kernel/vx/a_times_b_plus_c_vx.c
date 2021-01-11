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
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"

#define REGISTER_A_TIMES_B_PLUS_C_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_A_TIMES_B_PLUS_C_OPENVX_KERNEL( a_times_b_plus_c )
{
    vx_node node = NULL;
    float scale;
    vsi_enum overflow_policy,rounding_policy;
    vx_scalar scale_s = NULL;
    vsi_nn_tensor_t * a_times_b = NULL;
    vsi_nn_tensor_attr_t attr;

    scale = 1.0;
    overflow_policy = VX_CONVERT_POLICY_SATURATE;
    rounding_policy = VX_ROUND_POLICY_TO_ZERO;

    scale_s = vxCreateScalar(graph->ctx->c, VX_TYPE_FLOAT32, &scale);
    if(!scale_s)
    {
        VSILOGE("CreateScalar fail\n");
        goto OnError;
    }

    memset(&attr, 0, sizeof(attr));
    memcpy(attr.size, outputs[0]->attr.size,  VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ));
    attr.dim_num = outputs[0]->attr.dim_num;
    attr.vtl = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    a_times_b = vsi_nn_CreateTensor(graph, &attr);

    node = vxTensorMultiplyNode( graph->g,
        inputs[0]->t, inputs[1]->t,
        scale_s,
        overflow_policy,
        rounding_policy,
        a_times_b->t );
    if( NULL == node )
    {
        VSILOGE("Call vxTensorMultiplyNode fail.(a_times_b_plus_c)");
        goto OnError;
    }

    node = vxTensorAddNode( graph->g, a_times_b->t, inputs[2]->t,
        VX_CONVERT_POLICY_SATURATE, outputs[0]->t );
    if( NULL == node )
    {
        VSILOGE("Call vxTensorAddNode fail.(a_times_b_plus_c)");
        goto OnError;
    }

OnError:
    if (scale_s) vxReleaseScalar(&scale_s);
    if (a_times_b) vsi_nn_ReleaseTensor(&a_times_b);

    return (vsi_nn_kernel_node_t)node;
} /* a_times_b_plus_c() */

#undef REGISTER_A_TIMES_B_PLUS_C_OPENVX_KERNEL

