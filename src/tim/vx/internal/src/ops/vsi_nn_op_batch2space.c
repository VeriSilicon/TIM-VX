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
#include "vsi_nn_test.h"
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
    vsi_nn_tensor_attr_t attr;
    memset(&param, 0, sizeof(vx_nn_reorg_params_ext_t));

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 2;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    block_size_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.batch2space.block_size,
        &attr);
    TEST_CHECK_PTR(block_size_tensor, final);

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 4;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    pad_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.batch2space.crop,
        &attr);
    TEST_CHECK_PTR(pad_tensor, final);

    param.base.block_size = REQUIRED_IO(block_size_tensor);
    param.pad = OPTIONAL_IO(pad_tensor);
    param.base.type = VX_REORG_BATCH_TO_SPACE_ND;
    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        (vx_nn_reorg_params_t *)&param,
        sizeof(vx_nn_reorg_params_ext_t),
        outputs[0]->t);

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

final:
    if (block_size_tensor) vsi_nn_ReleaseTensor(&block_size_tensor);
    if (pad_tensor) vsi_nn_ReleaseTensor(&pad_tensor);

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

    if (inputs[0]->attr.dim_num != 4)
    {
        VSILOGE("batch2space only support 4D");
        return FALSE;
    }

    if (self->nn_param.batch2space.block_size[0] < 0
        || self->nn_param.batch2space.block_size[1] < 0)
    {
        VSILOGE("Block size can't be less than zero in batch to space");
        return FALSE;
    }

    ret = vsi_nn_OpCheck(VSI_NN_OP_STRIDED_SLICE, self, inputs, outputs);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_batch2space_param * p;
    p = (vsi_nn_batch2space_param *)&(self->nn_param.batch2space);

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[3] =
            inputs[0]->attr.size[3] / p->block_size[0] / p->block_size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[1] =
            inputs[0]->attr.size[1] * p->block_size[1] - p->crop[2] - p->crop[3];
        outputs[0]->attr.size[0] =
            inputs[0]->attr.size[0] * p->block_size[0] - p->crop[0] - p->crop[1];
        outputs[0]->attr.dim_num = 4;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.batch2space.local.block_size_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.batch2space.local.block_size_tensor));
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BATCH2SPACE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
