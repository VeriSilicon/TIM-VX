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
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#define REGISTER_SOFTMAX_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_SOFTMAX_OPENVX_KERNEL( softmax )
{
    vx_node node = NULL;
    float beta = vsi_nn_kernel_param_get_float32(params, "beta");
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    int32_t axis = vsi_nn_kernel_param_get_int32(params, "axis");
    int32_t new_axis = 0;
    size_t size = sizeof(vx_nn_softmax_params_t);
#ifdef VX_SOFTMAX_AXIS_PARAMETER_SUPPORT
    vx_nn_softmax_params_ext_t paramExt;
    vx_nn_softmax_params_t *param = (vx_nn_softmax_params_t *)&paramExt;
    paramExt.base.beta = beta;
    paramExt.axis = axis;
    size = sizeof(vx_nn_softmax_params_ext_t);
#else
    vx_nn_softmax_params_t base;
    vx_nn_softmax_params_t *param = &base;

    memset(&base, 0, sizeof(vx_nn_softmax_params_t));
    base.beta = beta;
#endif

    vsi_nn_kernel_optimize_softmax_shape(
       inputs[0]->attr.size, inputs[0]->attr.dim_num, axis,
       shapes[0], &rank_in, &new_axis);

    if (new_axis == 1)
    {
        int32_t i = 0;
        new_axis ++;
        rank_in ++;
        for (i = rank_in - 1; i > 1; i--)
        {
            shapes[0][i] = shapes[0][i - 1];
        }
        shapes[0][1] = 1;
    }

#ifdef VX_SOFTMAX_AXIS_PARAMETER_SUPPORT
    paramExt.axis = new_axis;
#endif

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], rank_in );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[0], rank_in );

    node = vxSoftmaxLayer2( graph->g,
        reshape_tensors[0]->t,
        param,
        size,
        reshape_tensors[1]->t);
    if( NULL == node )
    {
        VSILOGE("Call vxSoftmaxLayer2 fail.(softmax)");
    }

    vsi_nn_ReleaseTensor( &reshape_tensors[0] );
    vsi_nn_ReleaseTensor( &reshape_tensors[1] );

    return (vsi_nn_kernel_node_t)node;
} /* softmax() */

#undef REGISTER_SOFTMAX_OPENVX_KERNEL
