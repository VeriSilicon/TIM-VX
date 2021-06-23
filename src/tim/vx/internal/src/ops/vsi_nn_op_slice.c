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
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_internal_node.h"
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
    if (self->input.num > 1)
    {
        vsi_status status = VSI_FAILURE;

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "slice",
            inputs, 2, outputs, _OUTPUT_NUM, NULL );

        if( self->n )
        {
            status = VSI_SUCCESS;
        }

        return status;
    }
    else
    {
        return vsi_nn_internal_compute_node( self );
    }
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    if (self->input.num > 1)
    {
        BEGIN_IO_TYPE_DECL(SLICE, 2, 1)
            IO_TYPE(D_F16,       D_I32,  D_F16)
            IO_TYPE(D_F16,       D_I32,  D_I8|Q_DFP)
            IO_TYPE(D_F16,       D_I32,  D_I16|Q_DFP)
            IO_TYPE(D_F16,       D_I32,  D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_I32,  D_F16)
            IO_TYPE(D_I16|Q_DFP, D_I32,  D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I32,  D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_I32,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I32,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I32,  D_U8|Q_ASYM)
            IO_TYPE(D_F32,       D_I32,  D_F32)
            IO_TYPE(D_I32,       D_I32,  D_I32)

            /* HW 9.0 */
            IO_TYPE(D_BF16,     D_I32,    D_BF16)
            END_IO_TYPE_DECL(SLICE)
            if (!VALIDATE_OP_IO_TYPES(SLICE, self, inputs, self->input.num, outputs, self->output.num))
            {
                char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
                VSILOGE("Inputs/Outputs data type not support: %s", desc);
                destroy_op_io_types_desc(desc);
                return FALSE;
            }
            return TRUE;
    }
    else
    {
        ret = vsi_nn_OpCheck(VSI_NN_OP_STRIDED_SLICE, self, inputs, outputs);
    }

    return ret;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if (self->input.num > 1)
    {
        return VSI_SUCCESS;
    }
    else
    {
        return vsi_nn_internal_optimize_node( self, direction );
    }
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_slice_param * p;
    vsi_nn_internal_node_t* curr = NULL;
    uint32_t i;

    if (self->nn_param.slice.dims == 0)
    {
        self->nn_param.slice.dims = inputs[0]->attr.dim_num;
    }

    vsi_nn_internal_init_node_wksp( self );
    p = (vsi_nn_slice_param *)&(self->nn_param.slice);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        for(i = 0; i < p->dims; i++)
        {
            outputs[0]->attr.size[i] = p->length[i];
        }
        outputs[0]->attr.dim_num = p->dims;
    }

    if (self->input.num > 1)
    {
        return TRUE;
    }

    for (i = 0; i < self->nn_param.slice.dims; i++)
    {
        p->lcl_data->begin_dims[i] = self->nn_param.slice.start[i];
        p->lcl_data->end_dims[i] = self->nn_param.slice.start[i] + self->nn_param.slice.length[i];
        p->lcl_data->stride_dims[i] = 1;
    }

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_STRIDED_SLICE, 0, 0 );
    curr->node->nn_param.strided_slice.begin_dims = p->lcl_data->begin_dims;
    curr->node->nn_param.strided_slice.begin_dims_num = inputs[0]->attr.dim_num;
    curr->node->nn_param.strided_slice.end_dims = p->lcl_data->end_dims;
    curr->node->nn_param.strided_slice.end_dims_num = inputs[0]->attr.dim_num;
    curr->node->nn_param.strided_slice.stride_dims = p->lcl_data->stride_dims;
    curr->node->nn_param.strided_slice.stride_dims_num = inputs[0]->attr.dim_num;
    curr->node->nn_param.strided_slice.begin_mask = 0;
    curr->node->nn_param.strided_slice.end_mask = 0;
    curr->node->nn_param.strided_slice.shrink_axis_mask = 0;
    curr->node->nn_param.strided_slice.new_axis_mask = 0;
    curr->inputs[0] = inputs[0];
    curr->outputs[0] = outputs[0];
    vsi_nn_internal_setup_node( self, curr );

    return TRUE;
} /* op_setup() */


static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_slice_param * p = NULL;

    p = &(self->nn_param.slice);

    p->lcl_data   =
        (vsi_nn_slice_lcl_data *)malloc(sizeof(vsi_nn_slice_lcl_data));
    if (NULL == p->lcl_data)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(p->lcl_data, 0, sizeof(vsi_nn_split_lcl_data));

    return status;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_slice_param * p = NULL;

    p = &(self->nn_param.slice);

    if (p->lcl_data)
    {
        free(p->lcl_data);
        p->lcl_data = NULL;
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
        /* op_name    */ SLICE,
        /* init       */ op_init,
        /* compute    */ op_compute,
        /* deinit     */ op_deinit,
        /* check      */ op_check,
        /* setup      */ op_setup,
        /* optimize   */ op_optimize,
        /* input_num  */ 1,
        /* output_num */ 1
        );
#ifdef __cplusplus
}
#endif
