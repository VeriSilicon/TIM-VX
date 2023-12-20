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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vx_nn_reorg_params_ext_t param;
    vsi_nn_tensor_t *block_size_tensor = NULL;
    vsi_nn_tensor_t *pad_tensor = NULL;
    vsi_nn_tensor_t *input_tensor = NULL;
    vsi_nn_tensor_t *output_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_bool need_release_tensor = TRUE;
    int32_t block_size[2] = {1, 1};

    block_size[0] = self->nn_param.space2batch.block_size[0];
    if (vsi_nn_is_3d_tensor(inputs[0]))
    {
        vsi_size_t shape[2][VSI_NN_MAX_DIM_NUM] = {{1}};
        memcpy(shape[0], inputs[0]->attr.size, sizeof(shape[0]));
        memcpy(shape[1], outputs[0]->attr.size, sizeof(shape[1]));
        shape[0][3] = shape[0][2];
        shape[0][2] = shape[0][1];
        shape[0][1] = 1;
        shape[1][3] = shape[1][2];
        shape[1][2] = shape[1][1];
        shape[1][1] = 1;

        input_tensor = vsi_nn_reshape_tensor(self->graph, inputs[0], shape[0], 4);
        CHECK_PTR_FAIL_GOTO( input_tensor, "craete tensor fail.", final );
        output_tensor = vsi_nn_reshape_tensor(self->graph, outputs[0], shape[1], 4);
        CHECK_PTR_FAIL_GOTO( output_tensor, "craete tensor fail.", final );
    }
    else
    {
        block_size[1] = self->nn_param.space2batch.block_size[1];
        need_release_tensor = FALSE;
        input_tensor = inputs[0];
        output_tensor = outputs[0];
    }

    memset(&param, 0, sizeof(vx_nn_reorg_params_t));
    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 2;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    block_size_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)block_size,
        &attr);
    CHECK_PTR_FAIL_GOTO( block_size_tensor, "craete tensor fail.", final );

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 4;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    pad_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.space2batch.pad,
        &attr);
    CHECK_PTR_FAIL_GOTO( pad_tensor, "craete tensor fail.", final );

    param.base.block_size = REQUIRED_IO(block_size_tensor);
    param.pad = OPTIONAL_IO(pad_tensor);
    param.base.type = VX_REORG_SPACE_TO_BATCH_ND;

    self->n = vxReorgLayer2( self->graph->g,
        input_tensor->t,
        (vx_nn_reorg_params_t *)&param,
        sizeof(vx_nn_reorg_params_ext_t),
        output_tensor->t);

    if ( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

final:
    if (need_release_tensor)
    {
        vsi_safe_release_tensor(input_tensor);
        vsi_safe_release_tensor(output_tensor);
    }
    vsi_safe_release_tensor(block_size_tensor);
    vsi_safe_release_tensor(pad_tensor);

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

    if (inputs[0]->attr.dim_num < 3)
    {
        VSILOGE("The input tensor shape must be 3D or 4D!(space2batch)");
        return FALSE;
    }

    if (self->nn_param.space2batch.block_size[0] < 0
        || self->nn_param.space2batch.pad[0] < 0
        || self->nn_param.space2batch.pad[1] < 0
        || self->nn_param.space2batch.pad[2] < 0
        || self->nn_param.space2batch.pad[3] < 0)
    {
        VSILOGE("Block size or pad can't be less than zero in space to batch");
        return FALSE;
    }

    ret = vsi_nn_OpCheck(VSI_NN_OP_STRIDED_SLICE, self, inputs, outputs);

    return ret;
} /* op_add_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_space2batch_param * p;
    p = (vsi_nn_space2batch_param *)&(self->nn_param.space2batch);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;

        if (vsi_nn_is_3d_tensor(inputs[0]))
        {
            outputs[0]->attr.size[2] =
                inputs[0]->attr.size[2] * p->block_size[0];
            outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
            outputs[0]->attr.size[0] =
                (p->pad[0] + p->pad[1] + inputs[0]->attr.size[0]) / p->block_size[0];
        }
        else
        {
            outputs[0]->attr.size[3] =
                inputs[0]->attr.size[3] * p->block_size[0] * p->block_size[1];
            outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
            outputs[0]->attr.size[1] =
                (p->pad[2] + p->pad[3] + inputs[0]->attr.size[1]) / p->block_size[1];
            outputs[0]->attr.size[0] =
                (p->pad[0] + p->pad[1] + inputs[0]->attr.size[0]) / p->block_size[0];
        }
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_space2batch_param *p = &self->nn_param.space2batch;

    memset(p->pad, 0, sizeof(p->pad));

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SPACE2BATCH,
    /* init       */ op_init,
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
