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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_kernel_node_t    n = NULL;
    float eps = self->nn_param.layernorm.eps;

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16 &&
        inputs[2]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 &&
        outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8 )
    {
        return vsi_nn_internal_compute_node( self );
    }

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_float32( param, "eps", eps );
    n = vsi_nn_kernel_selector( self->graph, "layer_norm",
                    inputs, _INPUT_NUM, outputs, _OUTPUT_NUM, param );
    if ( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
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
    vsi_bool ret = TRUE;
    vsi_nn_internal_node_t* curr = NULL;

    if ( NULL == self )
    {
        return FALSE;
    }

    vsi_nn_internal_init_node_wksp( self );

   if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16 &&
        inputs[2]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 &&
        outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8 )
    {
        vsi_nn_internal_tensor_t* mean_tensor = NULL;
        vsi_nn_internal_tensor_t* vari_tensor = NULL;
        vsi_nn_tensor_attr_t attr;
        int32_t *axis = NULL;

        memcpy( &attr, &inputs[0]->attr, sizeof( attr ) );
        attr.size[0] = 1;
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        mean_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        vari_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_MOMENTS, 0, 0);
        axis = (int32_t*)\
                    vsi_nn_internal_new_node_param(curr, sizeof(int32_t) * 4);
        axis[0] = 0;

        curr->node->nn_param.moments.axis = axis;
        curr->node->nn_param.moments.axis_num = 1;
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = mean_tensor->t;
        curr->outputs[1] = vari_tensor->t;
        vsi_nn_internal_setup_node( self, curr );

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_BATCHNORM_SINGLE, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->inputs[1] = mean_tensor->t;
        curr->inputs[2] = vari_tensor->t;
        curr->inputs[3] = inputs[2];
        curr->inputs[4] = inputs[1];
        curr->node->nn_param.batchnorm_single.eps = self->nn_param.layernorm.eps;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node( self, curr );
    }
    else
    {
        ret = vsi_nn_op_common_setup(self, inputs, outputs);
    }

    return ret;
}

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(LAYER_NORM, 3, 1)
        IO_TYPE(D_F32,  D_F32,  D_F32,  D_F32)
        IO_TYPE(D_F16,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_F16,  D_F32,  D_F32,  D_F16)
        IO_TYPE(D_F16,  D_F32,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_F32,  D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_BF16, D_F32,  D_F32,  D_BF16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F16,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F32,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F32,  D_I16|Q_DFP)
    END_IO_TYPE_DECL(LAYER_NORM)
    if (!VALIDATE_OP_IO_TYPES(LAYER_NORM, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i = 0;
    for (i = 0; i < _VSI_NN_LAYERNORM_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.layernorm.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.layernorm.local.local_tensor[i]));
            self->nn_param.layernorm.local.local_tensor[i] = NULL;
        }
    }

    vsi_nn_internal_deinit_node_wksp( self );

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LAYER_NORM,
    /* init       */ NULL,
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
