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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (2)
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
    uint32_t i = 0;
    uint32_t block_size = 1, block_num = 1, axis_num = 0, indices_num = 1;
    int32_t axis = self->nn_param.gather.axis;
    uint32_t *input_size = inputs[0]->attr.size;
    uint32_t dims_num = inputs[0]->attr.dim_num;

    param =vsi_nn_kernel_param_create();

    for(i = 0; i < (uint32_t)axis; ++i)
    {
        block_size *= input_size[i];
    }

    axis_num = input_size[axis];
    for(i = axis + 1; i < dims_num; ++i)
    {
        block_num *= input_size[i];
    }
    for(i = 0; i < (uint32_t)inputs[1]->attr.dim_num; ++i)
    {
        indices_num *= inputs[1]->attr.size[i];
    }

    vsi_nn_kernel_param_add_int32( param, "block_size", block_size );
    vsi_nn_kernel_param_add_int32( param, "block_num", block_num );
    vsi_nn_kernel_param_add_int32( param, "axis_num", axis_num );
    vsi_nn_kernel_param_add_int32( param, "axis", axis );
    vsi_nn_kernel_param_add_int32( param, "indices_num", indices_num );
    n = vsi_nn_kernel_selector( self->graph, "gather", inputs, 2, outputs, 1, param );
    if( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if(param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
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
    BEGIN_IO_TYPE_DECL(GATHER, 2, 1)
        IO_TYPE(D_I32,  D_I32,  D_I32)
        IO_TYPE(D_F32,  D_I32,  D_F32)
        IO_TYPE(D_F16, D_I32, D_U8|Q_ASYM)
        IO_TYPE(D_F16, D_I32, D_I16|Q_DFP)
        IO_TYPE(D_F16, D_I32, D_I8|Q_DFP)
        IO_TYPE(D_F16, D_I32, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_I32, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_I32, D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_I32,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I32,  D_F16)
        IO_TYPE(D_I16|Q_DFP, D_I32, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP, D_I32, D_F16)
    END_IO_TYPE_DECL(GATHER)
    if(!VALIDATE_OP_IO_TYPES(GATHER, self, inputs, self->input.num, outputs, self->output.num)) {
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
    /* TODO: Add code to comput outputs' shape. */
    uint32_t i = 0;
    vsi_nn_gather_param * p = NULL;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t j = 0;
        p = &(self->nn_param.gather);
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num + inputs[1]->attr.dim_num - 1;
        for (i = 0; i < (uint32_t)p->axis; i++)
        {
            outputs[0]->attr.size[j] = inputs[0]->attr.size[i];
            j++;
        }
        for (i = 0; i < inputs[1]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[j] = inputs[1]->attr.size[i];
            j++;
        }
        for (i = (uint32_t)p->axis + 1; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[j] = inputs[0]->attr.size[i];
            j++;
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_GATHER_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.gather.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.gather.local.local_tensor[i]));
            self->nn_param.gather.local.local_tensor[i] = NULL;
        }
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
    /* op_name    */ GATHER,
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
