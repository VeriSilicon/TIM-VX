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

#if (VX_ACTIVATION_EXT_SUPPORT)

#define REGISTER_SWISH_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_SWISH_OPENVX_KERNEL( swish )
{
    vx_node node = NULL;
    vsi_nn_swish_type swish_type  = VSI_NN_SWISH;
    vx_enum function = VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SWISH;
    float   beta        = 1.0f;

    if (VSI_NN_HW_EVIS_2 == graph->ctx->config.evis.ver)
    {
        swish_type = (vsi_nn_swish_type)vsi_nn_kernel_param_get_int32(params, "type");

        if (VSI_NN_SWISH == swish_type)
        {
            function = VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SWISH;
        }
        else
        {
            function = VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HWISH;
        }

        beta = vsi_nn_kernel_param_get_float32( params, "beta" );

        node = vxActivationLayer(
            graph->g,
            inputs[0]->t,
            function,
            1,
            beta,
            outputs[0]->t
            );
    }
    return (vsi_nn_kernel_node_t)node;
} /* prelu() */

#undef REGISTER_SWISH_OPENVX_KERNEL

#endif
