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

#define _INPUT_NUM    1
#define _OUTPUT_NUM   2

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
    int32_t* axis = self->nn_param.moments.axis;
    int32_t axis_num = self->nn_param.moments.axis_num;
    int32_t keep_dim = self->nn_param.moments.keep_dim ? 1 : 0;

    param =vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_buffer( param, "axis", axis, axis_num);
    vsi_nn_kernel_param_add_int32( param, "keep_dim", keep_dim);
    n = vsi_nn_kernel_selector( self->graph, "moments", inputs, _INPUT_NUM, outputs, _OUTPUT_NUM, param );
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
    BEGIN_IO_TYPE_DECL(MOMENTS, 1, 2)
        IO_TYPE(D_U8|Q_ASYM,  D_F16,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_F16,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F16,  D_F16)
        IO_TYPE(D_F16,   D_F16,  D_F16)
        IO_TYPE(D_F32,   D_F32,  D_F32)
        IO_TYPE(D_I32,   D_F32,  D_F32)
    END_IO_TYPE_DECL(MOMENTS)
    if(!VALIDATE_OP_IO_TYPES(MOMENTS, self, inputs, self->input.num, outputs, self->output.num)) {
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
    int32_t i = 0, j = 0;
    vsi_nn_moments_param * p = NULL;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        int32_t* axis = NULL;
        int32_t axis_num = 0;
        p = &(self->nn_param.moments);
        axis = p->axis;
        axis_num = p->axis_num;

        for(i = 0; i < axis_num; i++)
        {
            if(axis[i] > 2)
            {
                return FALSE;
            }
        }

        if(p->keep_dim)
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
            outputs[1]->attr.dim_num = inputs[0]->attr.dim_num;

            for (i = 0; i < (int32_t)inputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
                outputs[1]->attr.size[i] = inputs[0]->attr.size[i];
            }
            switch(axis_num)
            {
            case 1:
                outputs[0]->attr.size[axis[0]] = 1;
                outputs[1]->attr.size[axis[0]] = 1;
                break;
            case 2:
                outputs[0]->attr.size[axis[0]] = 1;
                outputs[0]->attr.size[axis[1]] = 1;
                outputs[1]->attr.size[axis[0]] = 1;
                outputs[1]->attr.size[axis[1]] = 1;
                break;
            case 3:
                outputs[0]->attr.size[axis[0]] = 1;
                outputs[0]->attr.size[axis[1]] = 1;
                outputs[0]->attr.size[axis[2]] = 1;
                outputs[1]->attr.size[axis[0]] = 1;
                outputs[1]->attr.size[axis[1]] = 1;
                outputs[1]->attr.size[axis[2]] = 1;
                break;
            default:
                return FALSE;
            }
        }
        else
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num - axis_num;
            outputs[1]->attr.dim_num = inputs[0]->attr.dim_num - axis_num;

            for (i = 0; i < axis[0]; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
                outputs[1]->attr.size[i] = inputs[0]->attr.size[i];
            }

            for (j = axis[0] + axis_num; j < (int32_t)inputs[0]->attr.dim_num; j++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[j];
                outputs[1]->attr.size[i++] = inputs[0]->attr.size[j];
            }
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MOMENTS,
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
