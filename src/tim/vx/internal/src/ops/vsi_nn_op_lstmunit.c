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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"

static vsi_status _create_local_tensor
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensor_t *act_tensor = NULL;
    vsi_nn_tensor_t *cell_clip_tensor = NULL;
    vsi_nn_tensor_t *proj_clip_tensor = NULL;
    vsi_nn_tensor_t *scratch_tensor = NULL;
    vsi_nn_tensor_t *forget_bias_tensor = NULL;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    act_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstmunit.activation,
        VSI_NN_TYPE_INT32);
    if(NULL == act_tensor)
    {
        goto error;
    }

    cell_clip_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstmunit.cell_clip,
        VSI_NN_TYPE_FLOAT32);
    if(NULL == cell_clip_tensor)
    {
        goto error;
    }

    proj_clip_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstmunit.proj_clip,
        VSI_NN_TYPE_FLOAT32);
    if(NULL == proj_clip_tensor)
    {
        goto error;
    }

    scratch_tensor = vsi_nn_CreateTensor( self->graph, &self->nn_param.lstmunit.local.scratch_attr );
    if(NULL == scratch_tensor)
    {
        goto error;
    }

    forget_bias_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstmunit.forget_bias,
        VSI_NN_TYPE_FLOAT32);
    if(NULL == forget_bias_tensor)
    {
        goto error;
    }

    self->nn_param.lstmunit.local.activation_tensor = act_tensor;
    self->nn_param.lstmunit.local.cell_clip_tensor = cell_clip_tensor;
    self->nn_param.lstmunit.local.proj_clip_tensor = proj_clip_tensor;
    self->nn_param.lstmunit.local.scratch_tensor = scratch_tensor;
    self->nn_param.lstmunit.local.forget_bias_tensor = forget_bias_tensor;
    return VSI_SUCCESS;
error:
    if(act_tensor)vsi_nn_ReleaseTensor(&act_tensor);
    if(cell_clip_tensor)vsi_nn_ReleaseTensor(&cell_clip_tensor);
    if(proj_clip_tensor)vsi_nn_ReleaseTensor(&proj_clip_tensor);
    if(scratch_tensor)vsi_nn_ReleaseTensor(&scratch_tensor);
    if(forget_bias_tensor)vsi_nn_ReleaseTensor(&forget_bias_tensor);
    return VSI_FAILURE;
} /* _create_local_tensor() */

static vsi_status _init_lstmunit_param
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vx_nn_lstm_params_ext_t *param
    )
{
    param->base.input2input_weight       = OPTIONAL_IO(inputs[3]);
    param->base.input2forget_weight      = REQUIRED_IO(inputs[4]);
    param->base.input2cell_weight        = REQUIRED_IO(inputs[5]);
    param->base.input2output_weight      = REQUIRED_IO(inputs[6]);

    param->base.recurrent2input_weight   = OPTIONAL_IO(inputs[7]);
    param->base.recurrent2forget_weight  = REQUIRED_IO(inputs[8]);
    param->base.recurrent2cell_weight    = REQUIRED_IO(inputs[9]);
    param->base.recurrent2output_weight  = REQUIRED_IO(inputs[10]);

    param->base.cell2input_weight        = OPTIONAL_IO(inputs[11]);
    param->base.cell2forget_weight       = OPTIONAL_IO(inputs[12]);
    param->base.cell2output_weight       = OPTIONAL_IO(inputs[13]);

    param->base.input_gate_bias          = OPTIONAL_IO(inputs[14]);
    param->base.forget_gate_bias         = REQUIRED_IO(inputs[15]);
    param->base.cell_bias                = REQUIRED_IO(inputs[16]);
    param->base.output_gate_bias         = REQUIRED_IO(inputs[17]);

    param->base.projection_weight        = OPTIONAL_IO(inputs[18]);
    param->base.projection_bias          = OPTIONAL_IO(inputs[19]);

    param->layernorm2input_weight        = OPTIONAL_IO(inputs[20]);
    param->layernorm2forget_weight       = OPTIONAL_IO(inputs[21]);
    param->layernorm2cell_weight         = OPTIONAL_IO(inputs[22]);
    param->layernorm2output_weight       = OPTIONAL_IO(inputs[23]);

    param->base.activation   = OPTIONAL_IO(self->nn_param.lstmunit.local.activation_tensor);
    param->base.cell_clip    = OPTIONAL_IO(self->nn_param.lstmunit.local.cell_clip_tensor);
    param->base.proj_clip    = OPTIONAL_IO(self->nn_param.lstmunit.local.proj_clip_tensor);

    param->forget_bias = REQUIRED_IO(self->nn_param.lstmunit.local.forget_bias_tensor);
    param->norm_gain = 1.0f;
    param->norm_shift = 0.0f;

    return VSI_SUCCESS;
} /* _init_lstmunit_param() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_lstm_params_ext_t param;

    status = VSI_FAILURE;
    memset(&param, 0, sizeof(param));

    status = _create_local_tensor(self);
    if(status != VSI_SUCCESS)
    {
        return status;
    }

    status = _init_lstmunit_param(self, inputs, &param);
    if(status != VSI_SUCCESS)
    {
        return status;
    }

    /* Support high precision for cell state input */
    if( inputs[2] != NULL && VSI_NN_TYPE_FLOAT32 == inputs[2]->attr.dtype.vx_type )
    {
        status = vsi_nn_SetTensorAttr(inputs[2], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
        if(VSI_SUCCESS != status)
        {
            VSILOGE("Set tensor attr of cell state input to high presision fail");
            return status;
        }
    }

    /* Support high precision for cell state output */
    if( outputs[2] != NULL && VSI_NN_TYPE_FLOAT32 == outputs[2]->attr.dtype.vx_type )
    {
        status = vsi_nn_SetTensorAttr(outputs[2], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
        if(VSI_SUCCESS != status)
        {
            VSILOGE("Set tensor attr of cell state output to high presision fail");
            return status;
        }
    }

    self->n = vxLstmUnitLayer(
                self->graph->g,
                REQUIRED_IO(inputs[0]),
                REQUIRED_IO(inputs[1]),
                REQUIRED_IO(inputs[2]),
                (vx_nn_lstm_params_t *)&param,
                sizeof(param),
                REQUIRED_IO(self->nn_param.lstmunit.local.scratch_tensor),
                REQUIRED_IO(outputs[1]),
                REQUIRED_IO(outputs[2]),
                REQUIRED_IO(outputs[0])
                );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
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
    //TODO: Check tensor shapes.
    return TRUE;
} /* op_check() */

/*
    inputs[0]: input
    inputs[1]: output_state_in
    inputs[2]: cell_state_in
    inputs[3] ~ inputs[23]: weights & bias
    outputs[0]: scratch
    outputs[1]: output_state_out
    outputs[2]: cell_state_out
    outputs[3]: output
*/
static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* scratch */
    self->nn_param.lstmunit.local.scratch_attr.vtl = TRUE;
    self->nn_param.lstmunit.local.scratch_attr.is_const = FALSE;
    self->nn_param.lstmunit.local.scratch_attr.dtype.vx_type = outputs[0]->attr.dtype.vx_type;
    self->nn_param.lstmunit.local.scratch_attr.dim_num = inputs[0]->attr.dim_num;
    self->nn_param.lstmunit.local.scratch_attr.size[0] = inputs[4]->attr.size[1] * 4; /* num_units * 4 */
    self->nn_param.lstmunit.local.scratch_attr.size[1] = inputs[0]->attr.size[1];     /* batch_size */

    /* output */
    if(VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        if(inputs[18]) /* enable projection_weight */
        {
            outputs[0]->attr.size[0] = inputs[18]->attr.size[1];    /* output_size */
        }
        else /* disable projection_weight */
        {
            outputs[0]->attr.size[0] = inputs[4]->attr.size[1];    /* num_units */
        }
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];        /* batch_size */
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }

    /* output_state_out */
    if(VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num)
    {
        outputs[1]->attr.dim_num = outputs[0]->attr.dim_num;
        memcpy( outputs[1]->attr.size, outputs[0]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
    }

    /* cell_state_out */
    if(VSI_NN_DIM_AUTO == outputs[2]->attr.dim_num)
    {
        outputs[2]->attr.dim_num = outputs[1]->attr.dim_num;
        outputs[2]->attr.size[0] = inputs[4]->attr.size[1]; /* num_units */
        outputs[2]->attr.size[1] = inputs[0]->attr.size[1]; /* batch_size */
    }

    if ((NULL != outputs[3]) && (NULL != inputs[4]))
    {
        uint32_t cifg_factor = /*input2input_weight*/inputs[3] == NULL ? 3/*use_cifg*/ : 4;
        outputs[3]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[3]->attr.size[0] = inputs[4]->attr.size[1] * cifg_factor;  /* num_units * 4 */
        outputs[3]->attr.size[1] = inputs[0]->attr.size[1]; /* batch_size */
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensor_t *activation_tensor, *cell_clip_tensor, *proj_clip_tensor;
    vsi_nn_tensor_t *scratch_tensor;
    vsi_nn_tensor_t *forget_bias_tensor;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    activation_tensor = self->nn_param.lstmunit.local.activation_tensor;
    cell_clip_tensor = self->nn_param.lstmunit.local.cell_clip_tensor;
    proj_clip_tensor = self->nn_param.lstmunit.local.proj_clip_tensor;
    scratch_tensor = self->nn_param.lstmunit.local.scratch_tensor;
    forget_bias_tensor = self->nn_param.lstmunit.local.forget_bias_tensor;
    if(NULL != self->n)
    {
        if(activation_tensor)vsi_nn_ReleaseTensor(&activation_tensor);
        if(cell_clip_tensor)vsi_nn_ReleaseTensor(&cell_clip_tensor);
        if(proj_clip_tensor)vsi_nn_ReleaseTensor(&proj_clip_tensor);
        if(scratch_tensor)vsi_nn_ReleaseTensor(&scratch_tensor);
        if(forget_bias_tensor)vsi_nn_ReleaseTensor(&forget_bias_tensor);
        vxReleaseNode( &self->n );
        self->n = NULL;
    }

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.lstmunit.activation = VSI_NN_ACT_TANH;

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTMUNIT,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 24,
    /* output_num */ 4
    );
#ifdef __cplusplus
}
#endif

