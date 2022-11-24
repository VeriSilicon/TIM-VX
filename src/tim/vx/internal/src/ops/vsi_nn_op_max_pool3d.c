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
#include "utils/vsi_nn_link_list.h"
#include "vsi_nn_internal_node.h"

typedef struct _max_pool3d_local_data_t {
    int32_t placeholder;
} max_pool3d_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

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
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpCheck(VSI_NN_OP_POOL, self, inputs, outputs);

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
    return vsi_nn_internal_optimize_node( self, direction );
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_max_pool3d_param *p = &(self->nn_param.max_pool3d);
    vsi_size_t ksize[_cnt_of_array(p->ksize)] = {0}, i = 0;
    vsi_size_t pad[_cnt_of_array(p->pad)] = {0};
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* input_tensor = NULL;
    vsi_nn_internal_tensor_t* pool2d_0_tensor = NULL;
    vsi_nn_internal_tensor_t* reshape_0_tensor = NULL;
    vsi_nn_internal_tensor_t* pool2d_1_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_size_t* reshape_input_size = NULL;
    vsi_size_t* reshape_pool_size = NULL;

    for (i = 0; i < _cnt_of_array(p->ksize); i++)
    {
        ksize[i] = p->ksize[i];
    }
    for (i = 0; i < _cnt_of_array(p->pad); i++)
    {
        pad[i] = p->pad[i];
    }

    vsi_nn_compute_padding_3d(
        inputs[0]->attr.size,
        ksize,
        p->stride,
        NULL,
        p->pad_type,
        pad
        );

    for (i = 0; i < _cnt_of_array(p->ksize); i++)
    {
        p->ksize[i] = (uint32_t)ksize[i];
    }

    for (i = 0; i < _cnt_of_array(p->pad); i++)
    {
        p->pad[i] = (uint32_t)pad[i];
    }

    /* Pooling */
    outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[0],
        p->ksize[0],
        &p->pad[0],
        p->stride[0],
        0,
        p->round_type
        );

    outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[1],
        p->ksize[1],
        &p->pad[2],
        p->stride[1],
        0,
        p->round_type
        );

    outputs[0]->attr.size[2] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[2],
        p->ksize[2],
        &p->pad[4],
        p->stride[2],
        0,
        p->round_type
        );

    for (i = 3; i < inputs[0]->attr.dim_num; i++)
    {
        outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
    }

    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;

    vsi_nn_internal_init_node_wksp( self );

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &inputs[0]->attr.dtype, TRUE);
    input_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
    pool2d_0_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_input_size = vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    reshape_input_size[0] = inputs[0]->attr.size[0];
    reshape_input_size[1] = inputs[0]->attr.size[1];
    reshape_input_size[2] = 1;
    for (i = 2; i < inputs[0]->attr.dim_num; i++)
    {
        reshape_input_size[2] *= inputs[0]->attr.size[i];
    }
    reshape_input_size[3] = 1;
    curr->node->nn_param.reshape2.size = reshape_input_size;
    curr->node->nn_param.reshape2.dim_num = 4;
    curr->inputs[0] = inputs[0];
    curr->outputs[0] = input_tensor->t;
    vsi_nn_internal_setup_node( self, curr );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_POOL, 0, 0 );
    curr->node->nn_param.pool.ksize[0] = p->ksize[0];
    curr->node->nn_param.pool.ksize[1] = p->ksize[1];
    curr->node->nn_param.pool.stride[0] = p->stride[0];
    curr->node->nn_param.pool.stride[1] = p->stride[1];
    curr->node->nn_param.pool.pad[0] = p->pad[0];
    curr->node->nn_param.pool.pad[1] = p->pad[1];
    curr->node->nn_param.pool.pad[2] = p->pad[2];
    curr->node->nn_param.pool.pad[3] = p->pad[3];
    curr->node->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    curr->node->nn_param.pool.round_type = p->round_type;
    curr->node->nn_param.pool.pad_type = p->pad_type;
    curr->inputs[0] = input_tensor->t;
    curr->outputs[0] = pool2d_0_tensor->t;
    vsi_nn_internal_setup_node( self, curr );

    if (p->ksize[2] == 1 && p->stride[2] == 1 && p->pad[4] == 0 && p->pad[5] == 0)
    {
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
        curr->node->nn_param.reshape2.size = outputs[0]->attr.size;
        curr->node->nn_param.reshape2.dim_num = outputs[0]->attr.dim_num;
        curr->inputs[0] = pool2d_0_tensor->t;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node( self, curr );
    }
    else
    {
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        vsi_nn_internal_init_tensor_attr(&attr, &inputs[0]->attr.dtype, TRUE);
        reshape_0_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        pool2d_1_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
        reshape_pool_size = vsi_nn_internal_new_node_param(curr,
            VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        reshape_pool_size[0] = -1;
        reshape_pool_size[1] = inputs[0]->attr.size[2];
        reshape_pool_size[2] = 1;
        for (i = 3; i < inputs[0]->attr.dim_num; i++)
        {
            reshape_pool_size[2] *= inputs[0]->attr.size[i];
        }
        reshape_pool_size[3] = 1;
        curr->node->nn_param.reshape2.size = reshape_pool_size;
        curr->node->nn_param.reshape2.dim_num = 4;
        curr->inputs[0] = pool2d_0_tensor->t;
        curr->outputs[0] = reshape_0_tensor->t;
        vsi_nn_internal_setup_node( self, curr );

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_POOL, 1, 1 );
        curr->node->nn_param.pool.ksize[0] = 1;
        curr->node->nn_param.pool.ksize[1] = p->ksize[2];
        curr->node->nn_param.pool.stride[0] = 1;
        curr->node->nn_param.pool.stride[1] = p->stride[2];
        curr->node->nn_param.pool.pad[0] = 0;
        curr->node->nn_param.pool.pad[1] = 0;
        curr->node->nn_param.pool.pad[2] = p->pad[4];
        curr->node->nn_param.pool.pad[3] = p->pad[5];
        curr->node->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
        curr->node->nn_param.pool.round_type = p->round_type;
        curr->node->nn_param.pool.pad_type = p->pad_type;
        curr->inputs[0] = reshape_0_tensor->t;
        curr->outputs[0] = pool2d_1_tensor->t;
        vsi_nn_internal_setup_node( self, curr );

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
        curr->node->nn_param.reshape2.size = outputs[0]->attr.size;
        curr->node->nn_param.reshape2.dim_num = outputs[0]->attr.dim_num;
        curr->inputs[0] = pool2d_1_tensor->t;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node( self, curr );
    }

    return ret;
} /* op_setup() */


static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );
    status = vsi_nn_op_common_deinit(self);

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MAX_POOL3D,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS
