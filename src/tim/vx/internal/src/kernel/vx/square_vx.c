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
    vsi_nn_kernel_t             * kernel
    )
{
    vx_node node = NULL;
#ifdef VX_USER_LOOKUP_TABLE_SUPPORT
    vx_lut lut1 = NULL;
    vx_lut lut2 = NULL;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_lut_params lut_param;

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32   ||
         outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32 )
    {
        return NULL;
    }

    lut_param.act_type = VSI_NN_KERNEL_LUT_SQUARE;

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
        node = vxActivationLayer(
            graph->g,
            inputs[0]->t,
            VX_NN_ACTIVATION_SQUARE,
            0,
            0,
            outputs[0]->t
            );
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
    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_NN_ACTIVATION_SQUARE,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
#endif
} /* _setup() */

#define REGISTER_SQUARE_OPENVX_KERNEL(KERNEL_NAME) \
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
                params, kernel); \
    } \
    REGISTER_BACKEND_OPENVX( KERNEL_NAME, _##KERNEL_NAME##_setup )

REGISTER_SQUARE_OPENVX_KERNEL( square )

#undef REGISTER_SQUARE_OPENVX_KERNEL
