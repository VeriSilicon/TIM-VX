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
    if(inputs[0]->t != NULL && outputs[0]->t != NULL &&
        self->nn_param.reshape2.local->initialized == FALSE)
    {
        self->n = vxTensorCopyNode(self->graph->g,
            inputs[0]->t, outputs[0]->t);
        if(NULL == self->n)
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
    //TODO: Check tensor shapes.
    return TRUE;
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
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
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
    vsi_bool ret;

    status = VSI_SUCCESS;
    ret = TRUE;
    if ( vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype) == FALSE)
    {
        return status;
    }

    if (self->nn_param.reshape2.local->initialized == FALSE)
    {
        VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
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
                ret = vsi_nn_ReshapeTensor( self->graph, inputs[0], outputs[0],
                    self->nn_param.reshape2.size, self->nn_param.reshape2.dim_num );
                if ( ret == FALSE )
                {
                    status = VSI_FAILURE;
                }
                self->nn_param.reshape2.local->initialized = TRUE;
            }
        }
    }

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
