
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
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*
        Need copy input data to output if don't reshape input to output
    */
    if(inputs[0]->t != NULL && outputs[0]->t != NULL &&
        self->nn_param.variable.local->initialized == FALSE)
    {
        self->n = vxTensorCopyNode(self->graph->g,
            inputs[0]->t, outputs[0]->t);
        if(NULL == self->n)
        {
            VSILOGE( "Create vxTensorCopyNode fail." );
            return VSI_FAILURE;
        }
        VSILOGD("Create a copy node for variable");
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
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpCheck(VSI_NN_OP_DATACONVERT, self, inputs, outputs);

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
    vsi_nn_variable_lcl_data *local = NULL;
    if( direction == VSI_NN_OPTIMIZE_BACKWARD )
    {
        return VSI_SUCCESS;
    }
    local = (vsi_nn_variable_lcl_data *)malloc(sizeof(vsi_nn_variable_lcl_data));
    if( NULL == local )
    {
        VSILOGE("malloc memory fail");
        return VSI_FAILURE;
    }
    memset(local, 0, sizeof(vsi_nn_variable_lcl_data));
    if( NULL != inputs[0]->t && NULL == outputs[0]->t &&
        vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype))
    {
        VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
#ifdef VSI_40BIT_VA_SUPPORT
        outputs[0]->t = vxReshapeTensor(inputs[0]->t, outputs[0]->attr.size, outputs[0]->attr.dim_num);
#else
        outputs[0]->t = vxReshapeTensor(inputs[0]->t, (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num);
#endif
        if( NULL == outputs[0]->t )
        {
            VSILOGE("Call vxReshapeTensor fail");
            free(local);
            local = NULL;
            return VSI_FAILURE;
        }
        local->initialized = TRUE;
    }
    else
    {
        local->initialized = FALSE;
    }
    self->nn_param.variable.local = local;
    return VSI_SUCCESS;
} /* op_optimize() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_variable_lcl_data *local = self->nn_param.variable.local;
    if(local)
    {
        free(local);
        local = NULL;
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
    /* op_name    */ VARIABLE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

