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

#if VX_BATCH_GEMM_API_SUPPORT

#define REGISTER_BATCH_GEMM_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_BATCH_GEMM_OPENVX_KERNEL( matrixmul )
{
    vx_node node = NULL;
    int32_t transposeA = vsi_nn_kernel_param_get_int32(params, "transposeA");
    int32_t transposeB = vsi_nn_kernel_param_get_int32(params, "transposeB");
    vx_scalar trans_a = vxCreateScalar(graph->ctx->c, VX_TYPE_BOOL, &transposeA);
    vx_scalar trans_b = vxCreateScalar(graph->ctx->c, VX_TYPE_BOOL, &transposeB);

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    node = vxBatchGemmNode(graph->g,
                      inputs[0]->t,
                      inputs[1]->t,
                      NULL,
                      trans_a,
                      trans_b,
                      NULL,
                      outputs[0]->t);

    if( NULL == node )
    {
        VSILOGI("Call vxBatchGemmNode fail.");
        goto OnError;
    }

OnError:
    if (trans_a) vxReleaseScalar(&trans_a);
    if (trans_b) vxReleaseScalar(&trans_b);

    return (vsi_nn_kernel_node_t)node;
} /* matrixmul() */

#undef REGISTER_BATCH_GEMM_OPENVX_KERNEL

#endif
