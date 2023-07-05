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

static vsi_nn_tensor_t * _reshape_to_1d_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input
    )
{
    vsi_nn_tensor_t *tensor = NULL;
    uint32_t i = 0;
    vsi_size_t size = 0;
    vsi_size_t shapes[VSI_NN_MAX_DIM_NUM] = { 1 };
    uint32_t one_rank = 0;

    for (i = 0; i < input->attr.dim_num; i++)
    {
        if (input->attr.size[i] != 1)
        {
            size = input->attr.size[i];
            one_rank ++;
        }
    }

    if (one_rank <= 1)
    {
        shapes[0] = size;
    }
    else
    {
        VSILOGD("Error: PRelu Driver API only support per-chanel \n");
        return NULL;
    }

    tensor = vsi_nn_reshape_tensor( graph, input, shapes, 1 );

    return tensor;
}

#define REGISTER_PRELU_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_PRELU_OPENVX_KERNEL( prelu )
{
    vsi_nn_tensor_t * alpha = NULL;
    vx_node node = NULL;
    int32_t is_per_channel_alpha = 0;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    is_per_channel_alpha = vsi_nn_kernel_param_get_int32(params, "is_per_channel_alpha");

    if (!is_per_channel_alpha)
    {
        return NULL;
    }

    alpha = _reshape_to_1d_tensor(graph, inputs[1]);

    node = vxPReluLayer( graph->g, inputs[0]->t, inputs[1]->t, outputs[0]->t );

    if (alpha)
    {
        vsi_nn_ReleaseTensor(&alpha);
        alpha = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
} /* prelu() */

#undef REGISTER_PRELU_OPENVX_KERNEL

