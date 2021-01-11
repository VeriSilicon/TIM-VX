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

    memset(&param, 0, sizeof(vx_nn_reorg_params_t));
    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 2;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    block_size_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.space2batch.block_size,
        &attr);
    if( NULL == block_size_tensor )
    {
        VSILOGE("Create block_size_tensor fail.(space2batch)");
        return VSI_FAILURE;
    }

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
    if( NULL == pad_tensor )
    {
        VSILOGE("Create pad_tensor fail.(space2batch)");
        vsi_nn_ReleaseTensor(&block_size_tensor);
        block_size_tensor = NULL;
        return VSI_FAILURE;
    }

    self->nn_param.space2batch.local.block_size_tensor = block_size_tensor;
    self->nn_param.space2batch.local.pad_tensor = pad_tensor;
    param.base.block_size = REQUIRED_IO(block_size_tensor);
    param.pad = OPTIONAL_IO(pad_tensor);
    param.base.type = VX_REORG_SPACE_TO_BATCH_ND;

    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        (vx_nn_reorg_params_t *)&param,
        sizeof(vx_nn_reorg_params_ext_t),
        outputs[0]->t);

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
    if (inputs[0]->attr.dim_num != 4)
    {
        VSILOGE("The input tensor shape must be 4-D!(space2batch)");
        return FALSE;
    }

    if(self->nn_param.space2batch.block_size[0] < 0
        || self->nn_param.space2batch.block_size[1] < 0
        || self->nn_param.space2batch.pad[0] < 0
        || self->nn_param.space2batch.pad[1] < 0
        || self->nn_param.space2batch.pad[2] < 0
        || self->nn_param.space2batch.pad[3] < 0)
    {
        VSILOGE("Block size or pad can't be less than zero in space to batch");
        return FALSE;
    }

    {
        BEGIN_IO_TYPE_DECL(SPACE2DEPTH, 1, 1)
            IO_TYPE(D_F16,  D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
            IO_TYPE(D_F32,  D_F32)
            IO_TYPE(D_F32,  D_BF16)
            IO_TYPE(D_BF16, D_F32)
        END_IO_TYPE_DECL(SPACE2DEPTH)
        if (!VALIDATE_OP_IO_TYPES(SPACE2DEPTH, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }
    return TRUE;
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

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[3] =
            inputs[0]->attr.size[3] * p->block_size[0] * p->block_size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[1] =
            (p->pad[2] + p->pad[3] + inputs[0]->attr.size[1]) / p->block_size[1];
        outputs[0]->attr.size[0] =
            (p->pad[0] + p->pad[1] + inputs[0]->attr.size[0]) / p->block_size[0];
        outputs[0]->attr.dim_num = 4;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.space2batch.local.block_size_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.space2batch.local.block_size_tensor));
    }
    if (self->nn_param.space2batch.local.pad_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.space2batch.local.pad_tensor));
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
    /* op_name    */ SPACE2BATCH,
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

