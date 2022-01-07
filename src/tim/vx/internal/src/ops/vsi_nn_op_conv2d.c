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
#include <string.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "stride_w", self->nn_param.conv2d.stride[0] );
    vsi_nn_kernel_param_add_int32( param, "stride_h", self->nn_param.conv2d.stride[1] );
    vsi_nn_kernel_param_add_int32( param, "pad_h_front", self->nn_param.conv2d.pad[2] );
    vsi_nn_kernel_param_add_int32( param, "pad_h_end", self->nn_param.conv2d.pad[3] );
    vsi_nn_kernel_param_add_int32( param, "pad_w_front", self->nn_param.conv2d.pad[0] );
    vsi_nn_kernel_param_add_int32( param, "pad_w_end", self->nn_param.conv2d.pad[1] );
    vsi_nn_kernel_param_add_int32( param, "dilation_w", self->nn_param.conv2d.dilation[0] );
    vsi_nn_kernel_param_add_int32( param, "dilation_h", self->nn_param.conv2d.dilation[1] );
    vsi_nn_kernel_param_add_int32( param, "overflow_policy", self->vx_param.overflow_policy );
    vsi_nn_kernel_param_add_int32( param, "rounding_policy", self->vx_param.rounding_policy );
    vsi_nn_kernel_param_add_int32( param,
            "down_scale_size_rounding", self->vx_param.down_scale_size_rounding );
    if (self->nn_param.conv2d.multiplier != 0) {
        vsi_nn_kernel_param_add_int32( param, "multiplier",
            self->nn_param.conv2d.multiplier );
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "depthwise_conv2d",
            inputs, 3, outputs, 1, param );
    } else {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "conv2d",
            inputs, 3, outputs, 1, param );
    }
    vsi_nn_kernel_param_release( &param );

   if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    /* Check fl and scale*/
    ret = vsi_nn_QuantCheck(inputs[0], inputs[1], inputs[2]);

    if(ret) {
        /* check inputs outputs data type */
        BEGIN_IO_TYPE_DECL(CONV2D, 3, 1)
            /* IO_TYPE(INPUT, WEIGHT, BIAS, OUTPUT) */
            IO_TYPE(D_F32, D_F32, D_F32, D_F32)
            IO_TYPE(D_F32, D_F32, D_F32, D_F16)

            IO_TYPE(D_F16, D_F16, D_F16, D_F16)
            IO_TYPE(D_F16, D_F16, D_F32, D_F16)

            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I32|Q_DFP, D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I32|Q_DFP, D_F16)

            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I32|Q_DFP, D_I16|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I64|Q_DFP, D_I16|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I64|Q_DFP, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP, D_I32|Q_ASYM, D_U8|Q_ASYM)

            IO_TYPE(D_BF16, D_BF16, D_F32, D_F32)
            IO_TYPE(D_BF16, D_BF16, D_F32, D_BF16)

            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_I8|Q_ASYM)
            IO_TYPE(D_I8|Q_SYM, D_I8|Q_SYM, D_I32|Q_SYM, D_I8|Q_SYM)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_I32|Q_SYM_PC, D_U8|Q_ASYM)

            /* IO_TYPE(INPUT, WEIGHT, NULL, OUTPUT) */
            IO_TYPE(D_F32, D_F32, D_NONE, D_F32)

            IO_TYPE(D_F16, D_F16, D_NONE, D_F16)

            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_NONE, D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_NONE, D_F16)

            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_NONE, D_I16|Q_DFP)

            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_NONE, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_NONE, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP, D_NONE, D_U8|Q_ASYM)

            IO_TYPE(D_BF16, D_BF16, D_NONE, D_F32)
            IO_TYPE(D_BF16, D_BF16, D_NONE, D_BF16)

            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_NONE, D_I8|Q_ASYM)
            IO_TYPE(D_I8|Q_SYM, D_I8|Q_SYM, D_NONE, D_I8|Q_SYM)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC, D_NONE, D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_NONE, D_U8|Q_ASYM)

            /* HW 9.0 */
            IO_TYPE(D_F32, D_BF16, D_F32, D_BF16)
            IO_TYPE(D_F32, D_BF16, D_NONE, D_BF16)
            /* HW 9.0.1 */
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I32|Q_DFP,     D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,     D_I64|Q_DFP,     D_F32)

            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)

            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_NONE,          D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_NONE,          D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,     D_I64|Q_DFP,     D_F32)

            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_NONE,          D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_U8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)

            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_NONE,          D_BF16)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_NONE,          D_F32)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I32|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I32|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I32|Q_DFP,     D_BF16)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I32|Q_DFP,     D_F32)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I64|Q_DFP,     D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I64|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I64|Q_DFP,     D_BF16)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP,    D_I64|Q_DFP,     D_F32)

            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_NONE,          D_BF16)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_NONE,          D_F32)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_I32|Q_SYM_PC,  D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_SYM_PC, D_I32|Q_SYM_PC,  D_F32)

            IO_TYPE(D_F16,       D_F16,          D_NONE,          D_BF16)
            IO_TYPE(D_F16,       D_F16,          D_NONE,          D_F32)
            IO_TYPE(D_F16,       D_F16,          D_F32,           D_BF16)
            IO_TYPE(D_F16,       D_F16,          D_F32,           D_F32)

            IO_TYPE(D_BF16,      D_BF16,         D_NONE,          D_F16)
            IO_TYPE(D_BF16,      D_BF16,         D_F32,           D_F16)

            IO_TYPE(D_F32,       D_BF16,         D_NONE,          D_F16)
            IO_TYPE(D_F32,       D_BF16,         D_NONE,          D_BF16)
            IO_TYPE(D_F32,       D_BF16,         D_NONE,          D_F32)
            IO_TYPE(D_F32,       D_BF16,         D_F32,           D_F16)
            IO_TYPE(D_F32,       D_BF16,         D_F32,           D_BF16)
            IO_TYPE(D_F32,       D_BF16,         D_F32,           D_F32)

            IO_TYPE(D_U4|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_U4|Q_ASYM)
            IO_TYPE(D_U4|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I4|Q_ASYM)
            IO_TYPE(D_U4|Q_ASYM, D_I8|Q_ASYM,    D_I32|Q_ASYM,    D_U4|Q_ASYM)
            IO_TYPE(D_U4|Q_ASYM, D_I8|Q_ASYM,    D_I32|Q_ASYM,    D_I4|Q_ASYM)
            IO_TYPE(D_U4|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_ASYM,    D_U4|Q_ASYM)
            IO_TYPE(D_U4|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_ASYM,    D_I4|Q_ASYM)
            IO_TYPE(D_I4|Q_ASYM, D_I8|Q_ASYM,    D_I32|Q_ASYM,    D_I4|Q_ASYM)
            IO_TYPE(D_I4|Q_ASYM, D_I8|Q_ASYM,    D_I32|Q_ASYM,    D_U4|Q_ASYM)
            IO_TYPE(D_I4|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_I4|Q_ASYM)
            IO_TYPE(D_I4|Q_ASYM, D_U8|Q_ASYM,    D_I32|Q_ASYM,    D_U4|Q_ASYM)
            IO_TYPE(D_I4|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I4|Q_ASYM)
            IO_TYPE(D_I4|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_U4|Q_ASYM)
            IO_TYPE(D_I4|Q_DFP,  D_I8|Q_DFP,     D_I32|Q_DFP,     D_I4|Q_DFP)

        END_IO_TYPE_DECL(CONV2D)
        ret = VALIDATE_OP_IO_TYPES(CONV2D, self, inputs, self->input.num, outputs, self->output.num);
        if(!ret) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }

        /* check parameters */
        if(inputs[1]->attr.size[0] * inputs[1]->attr.size[1] > 6400) {
            VSILOGE("Kernel size should <= 6400.");
            return FALSE;
        }
    }

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_conv2d_param *nn_param;
    vsi_size_t perm[] = { 3, 2, 0, 1 };
    vsi_size_t i, pad[_cnt_of_array(self->nn_param.conv2d.pad)] = {0};

    for(i = 0; i < _cnt_of_array(self->nn_param.conv2d.pad); i++)
    {
        pad[i] = self->nn_param.conv2d.pad[i];
    }
    /* TODO: Driver should handle this,
    * Check transpose
    * */
    if( VSI_NN_DIM_FMT_NHWC == inputs[1]->attr.dtype.fmt &&
        VSI_NN_TYPE_VDATA != inputs[1]->attr.dtype.vx_type )
    {
        vsi_nn_TransposeTensor( self->graph, inputs[1], perm, 4, NULL );
        inputs[1]->attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    }

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    nn_param = &self->nn_param.conv2d;

    vsi_nn_compute_padding(
        inputs[0]->attr.size,
        inputs[1]->attr.size,
        self->nn_param.conv2d.stride,
        self->nn_param.conv2d.dilation,
        self->nn_param.conv2d.pad_type,
        pad
    );
    for(i = 0; i < _cnt_of_array(self->nn_param.conv2d.pad); i++)
    {
        self->nn_param.conv2d.pad[i] = (uint32_t)pad[i];
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            inputs[1]->attr.size[0],
            &nn_param->pad[0],
            nn_param->stride[0],
            nn_param->dilation[0],
            VSI_NN_ROUND_FLOOR
            );
        outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[1],
            inputs[1]->attr.size[1],
            &nn_param->pad[2],
            nn_param->stride[1],
            nn_param->dilation[1],
            VSI_NN_ROUND_FLOOR
            );
        if(self->nn_param.conv2d.weights > 0)
        {
            outputs[0]->attr.size[2] = self->nn_param.conv2d.weights;
        }
        else if(self->nn_param.conv2d.multiplier > 0)
        {
            outputs[0]->attr.size[2] = inputs[0]->attr.size[2] * self->nn_param.conv2d.multiplier;
        }
        else
        {
            outputs[0]->attr.size[2] = inputs[1]->attr.size[3];
        }
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONV2D,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
