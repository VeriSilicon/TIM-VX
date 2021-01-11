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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_dtype_util.h"
#include "vsi_nn_internal_node.h"

extern vx_kernel_description_t * vx_kernel_PRE_PROCESS_TENSOR_list[];


static vsi_bool _is_same_type
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if(vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    uint32_t i;
    uint32_t axis;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_bool use_virtual_tensor = TRUE;

    vsi_nn_internal_init_node_wksp( self );

    if( self->nn_param.pre_process_tensor.dim_num != inputs[0]->attr.dim_num )
    {
        VSILOGE( "Error permute dims '%u' vs '%u' ",
            self->nn_param.permute.dim_num, inputs[0]->attr.dim_num );
        return FALSE;
    }

    ret = TRUE;
    /* output */
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        for( i = 0; i < self->nn_param.pre_process_tensor.dim_num; i ++ )
        {
            axis = self->nn_param.pre_process_tensor.perm[i];
            if( axis >= inputs[0]->attr.dim_num )
            {
                VSILOGE( "Error permute axis '%u', the dim is '%u' ",
                    axis, inputs[0]->attr.dim_num );
                ret = FALSE;
                break;
            }
            outputs[0]->attr.size[i] = inputs[0]->attr.size[axis];
        }
    }

    for (i = 0; i < self->nn_param.pre_process_tensor.dim_num; i++)
    {
        axis = self->nn_param.pre_process_tensor.perm[i];
        if (axis != i)
            break;
    }

    if (i == self->nn_param.pre_process_tensor.dim_num)
        self->nn_param.pre_process_tensor.local.enable_perm = FALSE;
    else
        self->nn_param.pre_process_tensor.local.enable_perm = TRUE;

    if (_is_same_type(self, inputs, outputs))
        self->nn_param.pre_process_tensor.local.enable_data_conv = FALSE;
    else
        self->nn_param.pre_process_tensor.local.enable_data_conv = TRUE;

    if (self->nn_param.pre_process_tensor.local.enable_data_conv == FALSE &&
        self->nn_param.pre_process_tensor.local.enable_perm == FALSE)
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
        curr->node->nn_param.reshape.size = outputs[0]->attr.size;
        curr->node->nn_param.reshape.dim_num = outputs[0]->attr.dim_num;
        curr->inputs[0] = inputs[PRE_PROCESS_TENSOR_INPUT];
        curr->outputs[0] = outputs[PRE_PROCESS_TENSOR_OUTPUT];

        vsi_nn_internal_setup_node(self, curr);
    }
    else if (self->nn_param.pre_process_tensor.local.enable_data_conv == TRUE &&
        self->nn_param.pre_process_tensor.local.enable_perm == FALSE)
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0] = inputs[PRE_PROCESS_TENSOR_INPUT];
        curr->outputs[0] = outputs[PRE_PROCESS_TENSOR_OUTPUT];

        vsi_nn_internal_setup_node(self, curr);
    }
    else if (self->nn_param.pre_process_tensor.local.enable_data_conv == FALSE &&
        self->nn_param.pre_process_tensor.local.enable_perm == TRUE)
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
        curr->node->nn_param.permute.perm = self->nn_param.pre_process_tensor.perm;
        curr->node->nn_param.permute.dim_num = self->nn_param.pre_process_tensor.dim_num;
        curr->inputs[0] = inputs[PRE_PROCESS_TENSOR_INPUT];
        curr->outputs[0] = outputs[PRE_PROCESS_TENSOR_OUTPUT];

        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        /* transpose to time_major */
        memcpy( &attr, &outputs[PRE_PROCESS_TENSOR_OUTPUT]->attr, sizeof( attr ) );
        memcpy( &attr.size, &inputs[PRE_PROCESS_TENSOR_INPUT]->attr.size, sizeof( attr.size ) );
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0] = inputs[PRE_PROCESS_TENSOR_INPUT];
        curr->outputs[0] = output_tensor->t;

        vsi_nn_internal_setup_node( self, curr );

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
        curr->node->nn_param.permute.perm = self->nn_param.pre_process_tensor.perm;
        curr->node->nn_param.permute.dim_num = self->nn_param.pre_process_tensor.dim_num;
        curr->inputs[0] = output_tensor->t;
        curr->outputs[0] = outputs[PRE_PROCESS_TENSOR_OUTPUT];

        vsi_nn_internal_setup_node(self, curr);
    }

    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );

    return status;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PRE_PROCESS_TENSOR,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ PRE_PROCESS_TENSOR_INPUT_CNT,
    /* output_num */ PRE_PROCESS_TENSOR_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
