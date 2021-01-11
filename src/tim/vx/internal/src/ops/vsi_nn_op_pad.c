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

#include <stdlib.h>
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_constraint_check.h"

vsi_status vsi_nn_InitPadParameter
    (
    vsi_nn_node_t * node,
    vx_nn_pad_params_t * param
    )
{
    int32_t pad_const_val;
    uint8_t i;
    if(NULL == node || NULL == param)
    {
        VSILOGE("Set param fail\n");
        return VSI_FAILURE;
    }

    memset(param, 0, sizeof(vx_nn_pad_params_t));
    pad_const_val = node->nn_param.pad.const_val;
    param->pad_mode = node->nn_param.pad.mode;
    param->pad_const = vxCreateScalar( node->graph->ctx->c, VX_TYPE_INT32, &pad_const_val );
    if( NULL == param->pad_const )
    {
        VSILOGE("Create scalar fail\n");
        return VSI_FAILURE;
    }
    switch (param->pad_mode)
    {
    case VSI_NN_PAD_MODE_CONSTANT:
        param->pad_mode = VX_PAD_CONSTANT;
        break;
    case VSI_NN_PAD_MODE_REPLICATE:
        param->pad_mode = VX_PAD_REPLICATE;
        break;
    case VSI_NN_PAD_MODE_SYMMETRIC:
        param->pad_mode = VX_PAD_MIRROR_SYMMETRIC;
        break;
    case VSI_NN_PAD_MODE_REFLECT:
        param->pad_mode = VX_PAD_MIRROR_REFLECT;
        break;
    default:
        VSILOGE("Wrong pad_mode value");
        break;
    }

    /*
     * work around(TODO):
     *      driver only support pad 2 dimensions
     */
    param->numViewDimensions = vsi_nn_max(node->nn_param.pad.dim_num, 2);
    param->pad_front_array = (int32_t *)malloc(sizeof(int32_t) * param->numViewDimensions);
    param->pad_back_array = (int32_t *)malloc(sizeof(int32_t) * param->numViewDimensions);
    memset(param->pad_front_array, 0, sizeof(int32_t) * param->numViewDimensions);
    memset(param->pad_back_array, 0, sizeof(int32_t) * param->numViewDimensions);
    for(i=0; i < vsi_nn_min(param->numViewDimensions, node->nn_param.pad.dim_num); i++)
    {
        param->pad_front_array[i] = (int32_t)node->nn_param.pad.front_size[i];
        param->pad_back_array[i]  = (int32_t)node->nn_param.pad.back_size[i];
    }

    return VSI_SUCCESS;
} /* vsi_nn_InitPadParameter() */

void vsi_nn_DeinitPadParameter
    (
    vx_nn_pad_params_t * param
    )
{
    if( NULL != param )
    {
        if( NULL != param->pad_const )
        {
            vxReleaseScalar( &param->pad_const );
        }
        if( NULL != param->pad_front_array )
        {
            free( param->pad_front_array );
        }
        if( NULL != param->pad_back_array )
        {
            free( param->pad_back_array );
        }
    }
} /* vsi_nn_DeinitPadParameter() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_pad_params_t p;

    status = VSI_FAILURE;
    if(VSI_SUCCESS != vsi_nn_InitPadParameter(self, &p))
    {
        VSILOGE("Set Pad Layer Parameter fail\n");
        return VSI_FAILURE;
    }

    self->n = vxTensorPadNode(
        self->graph->g,
        inputs[0]->t,
        outputs[0]->t,
        &p,
        sizeof(p)
        );

    vsi_nn_DeinitPadParameter(&p);

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
    BEGIN_IO_TYPE_DECL(PAD, 1, 1)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_F32,  D_BF16)
        IO_TYPE(D_BF16, D_F32)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
    END_IO_TYPE_DECL(PAD)
    if (!VALIDATE_OP_IO_TYPES(PAD, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    if(self->nn_param.pad.dim_num != inputs[0]->attr.dim_num
        && self->nn_param.pad.dim_num != 0 )
    {
        VSILOGE("Error:input tensor dim should be equal with pad's.");
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
    uint32_t i;
    if(self->nn_param.pad.dim_num == 0)
    {
        self->nn_param.pad.dim_num = (uint8_t)inputs[0]->attr.dim_num;
    }
    if(VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        for(i=0; i<self->nn_param.pad.dim_num; i++)
        {
            uint32_t front = self->nn_param.pad.front_size[i];
            uint32_t back  = self->nn_param.pad.back_size[i];
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i] + front + back;
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    else
    {
        for(i=0; i<self->nn_param.pad.dim_num; i++)
        {
            uint32_t front = self->nn_param.pad.front_size[i];
            uint32_t back  = self->nn_param.pad.back_size[i];

            if (front + back + inputs[0]->attr.size[i] != outputs[0]->attr.size[i])
            {
                VSILOGE("Error:output shape[%u] not equal front padding[%u] + input shape[%u] + back padding[%u]",
                    outputs[0]->attr.size[i], front, back);
                return FALSE;
            }
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
        /* op_name    */ PAD,
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
