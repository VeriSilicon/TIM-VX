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
#include "vsi_nn_test.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

typedef struct _shape_local_data_t {
    vsi_nn_tensor_t *shape_tensor;
} shape_local_data_t;

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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);

    if (outputs[0]->attr.dtype.vx_type != VSI_NN_TYPE_INT32)
    {
        VSILOGD("Outputs data type not support");

        return FALSE;
    }

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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vx_int32 shapes[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t i = 0;

    for ( i = 0; i < inputs[0]->attr.dim_num; i++ )
    {
        shapes[i] = (int32_t)inputs[0]->attr.size[i];
    }

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = inputs[0]->attr.dim_num;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    self->nn_param.shape.local->shape_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)shapes,
        &attr);
    if ( NULL == self->nn_param.shape.local->shape_tensor )
    {
        VSILOGE("Create shape_tensor fail.(shape)");
        goto final;
    }

    vsi_nn_internal_init_node_wksp(self);
    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_DATACONVERT, 1, 1);
    if (NULL == curr)
    {
        return ret;
    }
    curr->inputs[0]  = self->nn_param.shape.local->shape_tensor;
    curr->outputs[0] = outputs[0];

    ret = vsi_nn_internal_setup_node(self, curr);

final:

    return ret;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    self->nn_param.shape.local =
        (shape_local_data_t *)malloc(sizeof(shape_local_data_t));
    if (NULL == self->nn_param.shape.local)
    {
        return VSI_FAILURE;
    }
    memset( self->nn_param.shape.local, 0, sizeof(shape_local_data_t) );

    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.shape.local)
    {
        vsi_safe_release_tensor(self->nn_param.shape.local->shape_tensor);
    }
    vsi_nn_safe_free(self->nn_param.shape.local);
    vsi_nn_internal_deinit_node_wksp(self);
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SHAPE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

