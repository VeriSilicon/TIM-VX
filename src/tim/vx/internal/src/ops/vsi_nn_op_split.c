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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_link_list.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_constraint_check.h"

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
    vsi_bool ret;
    uint32_t num,i,j;
    uint32_t slices_num = self->nn_param.split.slices_num;
    uint32_t axis = self->nn_param.split.axis;

    /* compute the output tensor number */
    num = (uint32_t)(self->output.num - 1);
    while ( NULL == outputs[num] )
    {
        num --;
    }
    num++;

    ret = TRUE;
    /* 1. check the input tensor number */
    if (self->input.num != 1)
    {
        VSILOGE("The split layer input num must be 1, here is %u\n", self->input.num);
        return FALSE;
    }

    /* 2. check output tensor number */
    if (slices_num == 0)
    {
        uint32_t remaind = inputs[0]->attr.size[axis] % num;
        if (remaind != 0)
        {
            VSILOGE("Can not average the input tensor %u shape\n", axis);
            return FALSE;
        }
    }
    else if (slices_num != num)
    {
        VSILOGE( "slices num %u != output tensor num %u\n", slices_num, num);
        return FALSE;
    }

    /* 3. check output tensor shape and dimensions */
    for ( i = 0; i < num; i ++ )
    {
        /* the virtual tensor shape has not been calculated yet */
        if (outputs[i]->attr.vtl == TRUE
            || outputs[i]->attr.dim_num == VSI_NN_DIM_AUTO)
            continue;

        if ( outputs[i]->attr.dim_num != inputs[0]->attr.dim_num )
        {
            VSILOGE( "Split dims num(%d vs %d)",
                outputs[i]->attr.dim_num,
                inputs[0]->attr.dim_num);
            ret = FALSE;
            break;
        }

        for ( j = 0; j < outputs[i]->attr.dim_num; j ++ )
        {
            if ( axis == j )
            {
                continue;
            }

            if ( outputs[i]->attr.size[j] != inputs[0]->attr.size[j] )
            {
                VSILOGE( "Split dims size(%d vs %d)",
                    outputs[i]->attr.size[j],
                    inputs[0]->attr.size[j]);
                ret = FALSE;
                break;
            }
        }

        if ( FALSE == ret )
        {
            break;
        }
    }
    for (i = 0; i < num; i++)
    {
        BEGIN_IO_TYPE_DECL(SPLIT, 1, 1)
            IO_TYPE(D_F16,  D_F16)
            IO_TYPE(D_F16,  D_I8|Q_DFP)
            IO_TYPE(D_F16,  D_I16|Q_DFP)
            IO_TYPE(D_F16,  D_U8|Q_ASYM)
            IO_TYPE(D_I8|Q_DFP,  D_F16)
            IO_TYPE(D_I16|Q_DFP, D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
            IO_TYPE(D_F16,  D_I8)
            IO_TYPE(D_F16,  D_I16)
            IO_TYPE(D_F16,  D_U8)
            IO_TYPE(D_I8,   D_F16)
            IO_TYPE(D_I16,  D_F16)
            IO_TYPE(D_U8,   D_F16)
            IO_TYPE(D_I8,   D_I8)
            IO_TYPE(D_I16,  D_I16)
            IO_TYPE(D_U8,   D_U8)
            IO_TYPE(D_F32,  D_F32)
            IO_TYPE(D_F32,  D_BF16)
            IO_TYPE(D_BF16, D_F32)
            IO_TYPE(D_I32,  D_I32)

            /* HW 9.0 */
            IO_TYPE(D_BF16, D_BF16)
        END_IO_TYPE_DECL(SPLIT)
        if (!VALIDATE_OP_IO_TYPES(SPLIT, self, inputs, 1, &outputs[i], 1)) {
            char* desc = generate_op_io_types_desc(inputs, 1, &outputs[i], 1);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }
    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    uint32_t i, num;
    vsi_size_t average;
    vsi_size_t start[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t end[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t axis = self->nn_param.split.axis;
    const uint32_t *slices = self->nn_param.split.slices;
    uint32_t slices_num = self->nn_param.split.slices_num;
    vsi_nn_split_param * p = NULL;
    vsi_nn_internal_node_t* curr = NULL;

    ret = TRUE;
    average = 1;
    /* compute the output tensor number */
    num = (uint32_t)(self->output.num - 1);
    while ( NULL == outputs[num] )
    {
        num --;
    }
    num++;

    p = &(self->nn_param.split);
    vsi_nn_internal_init_node_wksp( self );

    if (slices_num == 0)
    {
        average = inputs[0]->attr.size[axis] / num;
    }

    for (i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        p->lcl_data->stride_dims[i] = 1;
    }
    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        end[i] = inputs[0]->attr.size[i];
    }
    end[axis] = 0;
    for (i = 0; i < num; i++)
    {
        int32_t j;
        start[axis] = end[axis];
        if (slices_num == 0)
            end[axis] += average;
        else
            end[axis] += slices[i];

        outputs[i]->attr.dim_num = inputs[0]->attr.dim_num;
        for (j = 0; j < VSI_NN_MAX_DIM_NUM; j++)
        {
            outputs[i]->attr.size[j] = inputs[0]->attr.size[j];
        }
        outputs[i]->attr.size[axis] = end[axis] - start[axis];
        for (j = 0; j < VSI_NN_MAX_DIM_NUM; j++)
        {
            p->lcl_data->begin_dims[j] = (int32_t)start[j];
            p->lcl_data->end_dims[j] = (int32_t)end[j];
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
        curr->outputs[0] = outputs[i];
        vsi_nn_internal_setup_node( self, curr );
    }

    return ret;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_split_param * p = NULL;

    p = &(self->nn_param.split);

    p->lcl_data   =
    (vsi_nn_split_lcl_data *)malloc(sizeof(vsi_nn_split_lcl_data));
    if (NULL == p->lcl_data)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(p->lcl_data, 0, sizeof(vsi_nn_split_lcl_data));

    p->lcl_data->begin_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == p->lcl_data->begin_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(p->lcl_data->begin_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    p->lcl_data->end_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == p->lcl_data->end_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(p->lcl_data->end_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    p->lcl_data->stride_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == p->lcl_data->stride_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(p->lcl_data->stride_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    return status;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_split_param * p = NULL;

    p = &(self->nn_param.split);

    if (p->lcl_data->begin_dims)
    {
        free(p->lcl_data->begin_dims);
        p->lcl_data->begin_dims = NULL;
    }

    if (p->lcl_data->end_dims)
    {
        free(p->lcl_data->end_dims);
        p->lcl_data->end_dims = NULL;
    }

    if (p->lcl_data->stride_dims)
    {
        free(p->lcl_data->stride_dims);
        p->lcl_data->stride_dims = NULL;
    }

    if (p->lcl_data)
    {
        free(p->lcl_data);
        p->lcl_data = NULL;
    }

    vsi_nn_internal_deinit_node_wksp( self );

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SPLIT,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 16
    );
#ifdef __cplusplus
}
#endif
