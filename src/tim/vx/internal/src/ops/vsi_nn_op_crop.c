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
#include "libnnext/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (3)
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
    vx_nn_stride_slice_params_t param;
    vsi_nn_tensor_t *begin_dims_tensor = NULL;
    vsi_nn_tensor_t *end_dims_tensor = NULL;
    vsi_nn_tensor_t *stride_dims_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    int32_t start[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t end[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t stride[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t i;

    memset(&param, 0, sizeof(vx_nn_stride_slice_params_t));

    for (i = 0; i < self->nn_param.crop.dims; i++)
    {
        start[i] = self->nn_param.crop.offset[i];
        end[i] = self->nn_param.crop.offset[i] + outputs[0]->attr.size[i];
        stride[i] = 1;
    }

    for (i = self->nn_param.crop.dims; i < inputs[0]->attr.dim_num; i++)
    {
        start[i] = 0;
        end[i] = outputs[0]->attr.size[i];
        stride[i] = 1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = inputs[0]->attr.dim_num;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    begin_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)start,
        &attr);
    if( NULL == begin_dims_tensor )
    {
        VSILOGE("Create begin_dims_tensor fail.(crop)");
        return VSI_FAILURE;
    }

    end_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)end,
        &attr);
    if( NULL == end_dims_tensor )
    {
        VSILOGE("Create end_dims_tensor fail.(crop)");
        status = VSI_FAILURE;
        goto OnError;
    }

    stride_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)stride,
        &attr);
    if( NULL == stride_dims_tensor )
    {
        VSILOGE("Create stride_dims_tensor fail.(crop)");
        status = VSI_FAILURE;
        goto OnError;
    }

    param.begin_dims = REQUIRED_IO(begin_dims_tensor);
    param.end_dims = REQUIRED_IO(end_dims_tensor);
    param.stride_dims = REQUIRED_IO(stride_dims_tensor);

    self->n = vxTensorStrideSliceNode(
        self->graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_stride_slice_params_t),
        outputs[0]->t
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
OnError:
    if (begin_dims_tensor) vsi_nn_ReleaseTensor(&begin_dims_tensor);
    if (end_dims_tensor) vsi_nn_ReleaseTensor(&end_dims_tensor);
    if (stride_dims_tensor) vsi_nn_ReleaseTensor(&stride_dims_tensor);
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(CROP, 1, 1)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
    END_IO_TYPE_DECL(CROP)
    if (!VALIDATE_OP_IO_TYPES(CROP, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_crop_param * p;
    int32_t i;
    p = (vsi_nn_crop_param *)&(self->nn_param.crop);
    if (p->axis >= (int32_t)inputs[0]->attr.dim_num)
    {
        VSILOGE("Invalid parameter: axis!\n");
        return FALSE;
    }

    if( VSI_NN_DIM_AUTO != outputs[0]->attr.dim_num )
    {
        return TRUE;
    }

    if (p->dims + p->axis == inputs[0]->attr.dim_num)
    {
        for(i = 0; i < p->axis; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
        for(i = p->axis; i < (int32_t)inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    else
    {
        if (p->dims == 1)
        {
            for(i = 0; i <= p->axis; i++)
            {
                outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
                p->offset[i] = p->offset[0];
            }
            for(i = p->axis + 1; i < (int32_t)inputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
            }
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        }
        else
        {
            VSILOGE("Invalid parameter: offset dims!\n");
            return FALSE;
        }
    }

    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CROP,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

