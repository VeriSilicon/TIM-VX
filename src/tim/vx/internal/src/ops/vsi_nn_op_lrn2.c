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
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

#define MAX_BATCH_COUNT   1024

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    int32_t axis = -1;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t innerSize = 1;
    uint32_t outerSize = 1;
    uint32_t axisSize  = 1;
    vx_tensor vx_input = NULL;
    vx_tensor vx_output = NULL;
    vx_tensor input = inputs[0]->t;
    vx_tensor output = outputs[0]->t;
    uint32_t i = 0;

#ifdef VX_NORMALIZATION_AXIS_PARAMETER_SUPPORT
    vx_nn_normalization_params_ext_t param;

    memset(&param, 0, sizeof(vx_nn_normalization_params_ext_t));
    axis = self->nn_param.lrn.axis;
    param.base.alpha = self->nn_param.lrn.alpha;
    param.base.beta = self->nn_param.lrn.beta;
    param.base.bias = self->nn_param.lrn.bias;
    param.base.norm_size = self->nn_param.lrn.size;
    param.base.type = self->nn_param.lrn.type;
    param.axis = axis;

    if (param.base.type == VX_NN_NORMALIZATION_ACROSS_MAPS && axis != 2)
    {
        axisSize  = inputs[0]->attr.size[axis];

        for (i = 0; i < (uint32_t)axis; i++)
        {
            innerSize *= inputs[0]->attr.size[i];
        }

        for (i = (uint32_t)(axis + 1); i < inputs[0]->attr.dim_num; i++)
        {
            outerSize *= inputs[0]->attr.size[i];
        }

        sizes[0] = innerSize;
        sizes[1] = 1;
        sizes[2] = axisSize;
        sizes[3] = outerSize;

        if(outerSize < MAX_BATCH_COUNT)
        {
            vx_input = vxReshapeTensor(inputs[0]->t, (int32_t *)sizes, vsi_nn_max(inputs[0]->attr.dim_num, 4));
            vx_output = vxReshapeTensor(outputs[0]->t, (int32_t *)sizes, vsi_nn_max(inputs[0]->attr.dim_num, 4));

            input = vx_input;
            output = vx_output;

            param.axis = 2;
        }
    }

    self->n = vxNormalizationLayer2( self->graph->g,
        input,
        (vx_nn_normalization_params_t*)&param,
        sizeof(vx_nn_normalization_params_ext_t),
        output);
#else
    vx_nn_normalization_params_t param;

    memset(&param, 0, sizeof(vx_nn_normalization_params_t));
    axis = self->nn_param.lrn.axis;
    param.alpha = self->nn_param.lrn.alpha;
    param.beta = self->nn_param.lrn.beta;
    param.bias = self->nn_param.lrn.bias;
    param.norm_size = self->nn_param.lrn.size;
    param.type = self->nn_param.lrn.type;

    if (param.type == VX_NN_NORMALIZATION_ACROSS_MAPS && axis != 2)
    {
        axisSize  = inputs[0]->attr.size[axis];

        for (i = 0; i < (uint32_t)axis; i++)
        {
            innerSize *= inputs[0]->attr.size[i];
        }

        for (i = (uint32_t)(axis + 1); i < inputs[0]->attr.dim_num; i++)
        {
            outerSize *= inputs[0]->attr.size[i];
        }

        sizes[0] = innerSize;
        sizes[1] = 1;
        sizes[2] = axisSize;
        sizes[3] = outerSize;

        vx_input = vxReshapeTensor(inputs[0]->t, (int32_t *)sizes, vsi_nn_max(inputs[0]->attr.dim_num, 4));
        vx_output = vxReshapeTensor(outputs[0]->t, (int32_t *)sizes, vsi_nn_max(inputs[0]->attr.dim_num, 4));

        input = vx_input;
        output = vx_output;
    }

    self->n = vxNormalizationLayer2( self->graph->g,
        input,
        &param,
        sizeof(vx_nn_normalization_params_t),
        output);
#endif

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    if (vx_input) vxReleaseTensor(&vx_input);
    if (vx_output) vxReleaseTensor(&vx_output);

    return status;
} /* op_compute() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 15) == -1)
    {
        self->nn_param.lrn.axis = 2;
    }

    return status;
} /* op_init() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    //TODO: Check tensor shapes.
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpCheck(VSI_NN_OP_LRN, self, inputs, outputs);

    return ret;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LRN2,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

