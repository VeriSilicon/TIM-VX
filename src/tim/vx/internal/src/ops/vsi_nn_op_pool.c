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
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_pooling_params_ext_t params;
    status = VSI_FAILURE;

    memset( &params, 0, sizeof( params ) );
    params.base.pool_type = self->nn_param.pool.type;
    params.base.pool_size_x = self->nn_param.pool.ksize[0];
    params.base.pool_size_y = self->nn_param.pool.ksize[1];
    params.base.pool_pad_x_left = self->nn_param.pool.pad[0];
    params.base.pool_pad_x_right = self->nn_param.pool.pad[1];
    params.base.pool_pad_y_top = self->nn_param.pool.pad[2];
    params.base.pool_pad_y_bottom = self->nn_param.pool.pad[3];
    params.base.rounding = self->vx_param.down_scale_size_rounding;
    params.stride_x = self->nn_param.pool.stride[0];
    params.stride_y = self->nn_param.pool.stride[1];

    self->n = vxPoolingLayer2(
        self->graph->g,
        inputs[0]->t,
        (vx_nn_pooling_params_t *)&params,
        sizeof( params ),
        outputs[0]->t
        );

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
    /* check inputs outputs data type */
    BEGIN_IO_TYPE_DECL(POOL, 1, 1)
        /* IO_TYPE(INPUT, OUTPUT) */
        IO_TYPE(D_F32, D_F32)
        IO_TYPE(D_F32, D_F16)
        IO_TYPE(D_F32, D_BF16)

        IO_TYPE(D_F16, D_F32)
        IO_TYPE(D_F16, D_F16)
        IO_TYPE(D_F16, D_U8|Q_ASYM)
        IO_TYPE(D_F16, D_I8|Q_DFP)
        IO_TYPE(D_F16, D_I16|Q_DFP)

        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_BF16, D_F32)

        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_F16)

        IO_TYPE(D_I8|Q_ASYM, D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM, D_F16)

        IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP, D_F16)

        IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP, D_F16)
    END_IO_TYPE_DECL(POOL)
    if(!VALIDATE_OP_IO_TYPES(POOL, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;

    ret = TRUE;
    vsi_nn_compute_padding(
        inputs[0]->attr.size,
        self->nn_param.pool.ksize,
        self->nn_param.pool.stride,
        NULL,
        self->nn_param.pool.pad_type,
        self->nn_param.pool.pad
    );

    /* Pooling */
    outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[0],
        self->nn_param.pool.ksize[0],
        &self->nn_param.pool.pad[0],
        self->nn_param.pool.stride[0],
        0,
        self->nn_param.pool.round_type
        );
    outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[1],
        self->nn_param.pool.ksize[1],
        &self->nn_param.pool.pad[2],
        self->nn_param.pool.stride[1],
        0,
        self->nn_param.pool.round_type
        );

    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
    outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    if( NULL != outputs[1] )
    {
        outputs[1]->attr.dim_num = outputs[0]->attr.dim_num;
        memcpy( outputs[1]->attr.size, outputs[0]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ POOL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

