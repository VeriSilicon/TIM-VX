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
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vx_nn_reorg_params_ext2_t param;
    vsi_nn_tensor_t *block_size_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    uint8_t data = 1;

    memset(&param, 0, sizeof(vx_nn_reorg_params_ext2_t));
    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 2;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    block_size_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        &data,
        &attr);
    if( NULL == block_size_tensor )
    {
        VSILOGE("Create block_size_tensor fail.(shufflechannel)");
        return VSI_FAILURE;
    }

    self->nn_param.shufflechannel.local->block_size_tensor = block_size_tensor;
    param.base.block_size = REQUIRED_IO(block_size_tensor);

    param.base.type = VX_REORG_SHUFFLE_CHANNEL;
    param.axis = &self->nn_param.shufflechannel.axis;
    param.num_group = &self->nn_param.shufflechannel.group_number;

    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        (vx_nn_reorg_params_t *)&param,
        sizeof(vx_nn_reorg_params_ext2_t),
        outputs[0]->t);

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
} /* op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    vsi_nn_shufflechannel_param *p = NULL;
    int32_t axis = 0;

    if( NULL == self )
    {
        return ret;
    }

    p = &(self->nn_param.shufflechannel);
    axis = p->axis;

    if (axis < 0)
    {
        axis = axis + inputs[0]->attr.dim_num;
        p->axis = axis;
    }

    if (p->axis < 0)
    {
        VSILOGD("shufflechannel Invalid Axis: %d", p->axis);
        return FALSE;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        memcpy( outputs[0]->attr.size, inputs[0]->attr.size,
            sizeof(uint32_t) * inputs[0]->attr.dim_num );
    }

    return TRUE;
} /* op_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_shufflechannel_param *p = NULL;
    int32_t axis = 0;
    int32_t dims = (int32_t)inputs[0]->attr.dim_num;
    int32_t num_group = 0;
    uint32_t *shape = inputs[0]->attr.size;

    p = &(self->nn_param.shufflechannel);
    axis = p->axis;
    num_group = p->group_number;

    if (axis > (dims - 1))
    {
        VSILOGE("Invalid Axis: %d, (SHUFFLECHANNEL) at [%s : %d]\n", axis, __FILE__, __LINE__);
        return FALSE;
    }
    if (shape[axis] % num_group)
    {
        VSILOGE("Invalid group_number: %d, (SHUFFLECHANNEL) at [%s : %d]\n", num_group, __FILE__, __LINE__);
        return FALSE;
    }

    {
        BEGIN_IO_TYPE_DECL(SHUFFLECHANNEL, 1, 1)
            IO_TYPE(D_F16,  D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
            IO_TYPE(D_F32,  D_F32)
            IO_TYPE(D_BF16, D_F32)
            IO_TYPE(D_F32,  D_BF16)
        END_IO_TYPE_DECL(SHUFFLECHANNEL)
        if(!VALIDATE_OP_IO_TYPES(SHUFFLECHANNEL, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return TRUE;
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_shufflechannel_lcl_data_t *local = NULL;
    vsi_nn_shufflechannel_param *p = NULL;

    p = &(self->nn_param.shufflechannel);
    self->nn_param.shufflechannel.axis = 2;
    local = (vsi_nn_shufflechannel_lcl_data_t *)malloc(sizeof(vsi_nn_shufflechannel_lcl_data_t));
    if (NULL == local)
    {
        VSILOGE("Malloc fail, (SHUFFLECHANNEL) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(local, 0, sizeof(vsi_nn_shufflechannel_lcl_data_t));
    p->local = local;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_shufflechannel_param *p = &(self->nn_param.shufflechannel);
    if (p->local)
    {
        if (p->local->input_tensor)
        {
            vxReleaseTensor(&(p->local->input_tensor));
            p->local->input_tensor = NULL;
        }
        if (p->local->output_tensor)
        {
            vxReleaseTensor(&(p->local->output_tensor));
            p->local->output_tensor = NULL;
        }
        if (p->local->block_size_tensor != NULL)
        {
            vsi_nn_ReleaseTensor(&(p->local->block_size_tensor));
        }

        vsi_nn_safe_free(p->local);
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
    /* op_name    */ SHUFFLECHANNEL,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

