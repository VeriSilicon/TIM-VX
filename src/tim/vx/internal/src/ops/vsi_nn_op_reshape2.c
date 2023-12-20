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
#include <stdio.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*
    *If reshape is un-initialized, we need add a tensorcopy
    * when input and output are initialized.
    */
    if (inputs[0]->t != NULL && outputs[0]->t != NULL &&
        self->nn_param.reshape2.local->initialized == FALSE)
    {
#ifdef VX_REMOVE_RESHAPE_SUPPORT
        vsi_nn_tensor_attr_t attr;
        vsi_nn_tensor_t *dims_tensor = NULL;
        vx_nn_reshape_params_t reshape_param;
        int32_t dims_data[VSI_NN_MAX_DIM_NUM] = {1};
        uint32_t i = 0;

        for (i = 0; i < self->nn_param.reshape2.dim_num; i++)
        {
            dims_data[i] = (int32_t)self->nn_param.reshape2.size[i];
        }

        memset(&attr, 0, sizeof(attr));
        attr.size[0] = self->nn_param.reshape2.dim_num;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        dims_tensor = vsi_nn_CreateTensorFromData(
            self->graph,
            (uint8_t *)dims_data,
            &attr);

        reshape_param.dims = REQUIRED_IO(dims_tensor);

        self->n = vxTensorReshapeNode(self->graph->g,
            inputs[0]->t, &reshape_param, sizeof(reshape_param), outputs[0]->t);
        vsi_safe_release_tensor(dims_tensor);
#else
        vsi_nn_tensor_t *tmp_tensor = NULL;
        tmp_tensor = vsi_nn_reshape_tensor( self->graph,
            outputs[0], inputs[0]->attr.size, inputs[0]->attr.dim_num );
        CHECK_PTR_FAIL_GOTO( tmp_tensor, "create tensor fail.", final );

        self->n = vxTensorCopyNode(self->graph->g,
            inputs[0]->t, tmp_tensor->t);

final:
        vsi_safe_release_tensor(tmp_tensor);
#endif
        if (NULL == self->n)
        {
            VSILOGE( "Create vxTensorCopyNode fail." );
            return VSI_FAILURE;
        }
        VSILOGD("Create a copy node for reshape");
    }

    return VSI_SUCCESS;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_OpCheck(VSI_NN_OP_DATACONVERT, self, inputs, outputs);
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    self->nn_param.reshape2.local   =
    (vsi_nn_reshape2_local_data *)malloc(sizeof(vsi_nn_reshape2_local_data));
    if (NULL == self->nn_param.reshape2.local)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(self->nn_param.reshape2.local, 0, sizeof(vsi_nn_reshape2_local_data));
    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.reshape2.local != NULL)
    {
        free(self->nn_param.reshape2.local);
        self->nn_param.reshape2.local = NULL;
    }

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = {0};
        memcpy(shape, self->nn_param.reshape2.size,
            sizeof(vsi_size_t) * self->nn_param.reshape2.dim_num);
        ret = vsi_nn_CalcReshapeTensor(inputs[0],
            outputs[0],
            shape,
            self->nn_param.reshape2.dim_num);
    }

    return ret;
} /* op_setup() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status status;

    status = VSI_SUCCESS;
#ifdef VX_REMOVE_RESHAPE_SUPPORT
    self->nn_param.reshape2.local->initialized = FALSE;
#else
    if ( vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype) == FALSE)
    {
        return status;
    }

    if ( direction == VSI_NN_OPTIMIZE_BACKWARD )
    {
        if (NULL == inputs[0]->t && NULL != outputs[0]->t)
        {
            inputs[0]->t = vsi_nn_safe_reshape_tensor( outputs[0]->t,
                (void*)inputs[0]->attr.size, (vsi_size_t)inputs[0]->attr.dim_num,
                sizeof(inputs[0]->attr.size[0]) );
            if ( inputs[0]->t == NULL )
            {
                status = VSI_FAILURE;
            }
            self->nn_param.reshape2.local->initialized = TRUE;
        }
    }
    else
    {
        if (NULL == outputs[0]->t)
        {
            if ( NULL == inputs[0]->t )
            {
                vsi_nn_TensorReinit( self->graph, inputs[0] );
            }

            outputs[0]->t = vsi_nn_safe_reshape_tensor( inputs[0]->t,
                (void*)outputs[0]->attr.size, (vsi_size_t)outputs[0]->attr.dim_num,
                sizeof(outputs[0]->attr.size[0]) );
            if ( outputs[0]->t == NULL )
            {
                status = VSI_FAILURE;
            }
            self->nn_param.reshape2.local->initialized = TRUE;
        }
    }
#endif
    return status;
} /* op_optimize() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RESHAPE2,
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
