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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"

static vsi_status _create_local_tensor
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensor_t *act_tensor = NULL;
    vsi_nn_tensor_t *forget_bias_tensor = NULL;
    vsi_nn_tensor_t *cell_clip_tensor = NULL;
    vsi_nn_tensor_t *proj_clip_tensor = NULL;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    act_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstm.activation,
        VSI_NN_TYPE_INT32);
    if(NULL == act_tensor)
    {
        goto error;
    }

    if (self->nn_param.lstm.forget_bias != 0.0 )
    {
        forget_bias_tensor = vsi_nn_VariableToTensor(self,
            (uint8_t *)&self->nn_param.lstm.forget_bias,
            VSI_NN_TYPE_FLOAT32);
        if(NULL == forget_bias_tensor)
        {
            goto error;
        }
    }

    cell_clip_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstm.cell_clip,
        VSI_NN_TYPE_FLOAT32);
    if(NULL == cell_clip_tensor)
    {
        goto error;
    }

    proj_clip_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lstm.proj_clip,
        VSI_NN_TYPE_FLOAT32);
    if(NULL == proj_clip_tensor)
    {
        goto error;
    }

    self->nn_param.lstm.local.activation_tensor = act_tensor;
    self->nn_param.lstm.local.forget_bias_tensor = forget_bias_tensor;
    self->nn_param.lstm.local.cell_clip_tensor = cell_clip_tensor;
    self->nn_param.lstm.local.proj_clip_tensor = proj_clip_tensor;
    return VSI_SUCCESS;
error:
    if(act_tensor)vsi_nn_ReleaseTensor(&act_tensor);
    if(forget_bias_tensor)vsi_nn_ReleaseTensor(&forget_bias_tensor);
    if(cell_clip_tensor)vsi_nn_ReleaseTensor(&cell_clip_tensor);
    if(proj_clip_tensor)vsi_nn_ReleaseTensor(&proj_clip_tensor);
    return VSI_FAILURE;
} /* _create_local_tensor() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
#if 1
    vx_nn_lstm_layer_params_ext_t p;
    memset( &p, 0, sizeof( vx_nn_lstm_layer_params_ext_t ));

    status = VSI_FAILURE;

    status = _create_local_tensor(self);
    if(status != VSI_SUCCESS)
    {
        return status;
    }

    p.lstm_param.base.input2input_weight       = REQUIRED_IO(inputs[3]);
    p.lstm_param.base.input2forget_weight      = REQUIRED_IO(inputs[4]);
    p.lstm_param.base.input2cell_weight        = REQUIRED_IO(inputs[5]);
    p.lstm_param.base.input2output_weight      = REQUIRED_IO(inputs[6]);
    p.lstm_param.base.recurrent2input_weight   = REQUIRED_IO(inputs[7]);
    p.lstm_param.base.recurrent2forget_weight  = REQUIRED_IO(inputs[8]);
    p.lstm_param.base.recurrent2cell_weight    = REQUIRED_IO(inputs[9]);
    p.lstm_param.base.recurrent2output_weight  = REQUIRED_IO(inputs[10]);
    p.lstm_param.base.input_gate_bias          = REQUIRED_IO(inputs[14]);
    p.lstm_param.base.forget_gate_bias         = REQUIRED_IO(inputs[15]);
    p.lstm_param.base.cell_bias                = REQUIRED_IO(inputs[16]);
    p.lstm_param.base.output_gate_bias         = OPTIONAL_IO(inputs[17]);
    p.lstm_param.base.projection_weight        = OPTIONAL_IO(inputs[18]);
    p.lstm_param.base.projection_bias          = OPTIONAL_IO(inputs[19]);

    p.lstm_param.base.activation = OPTIONAL_IO(self->nn_param.lstm.local.activation_tensor);
    p.lstm_param.forget_bias = OPTIONAL_IO(self->nn_param.lstm.local.forget_bias_tensor);
    p.lstm_param.base.cell_clip  = REQUIRED_IO(self->nn_param.lstm.local.cell_clip_tensor);
    p.lstm_param.base.proj_clip  = REQUIRED_IO(self->nn_param.lstm.local.proj_clip_tensor);

    self->n = vxLstmLayer(
        self->graph->g,
        REQUIRED_IO(inputs[0]),
        NULL,
        NULL,
        ( vx_nn_lstm_layer_params_t *)&p,
        sizeof( vx_nn_lstm_layer_params_ext_t ),
        REQUIRED_IO(outputs[0])
        );
#else
    vx_nn_lstm_layer_params_t p;
    memset( &p, 0, sizeof( vx_nn_lstm_layer_params_t ));

    status = VSI_FAILURE;

    status = _create_local_tensor(self);
    if(status != VSI_SUCCESS)
    {
        return status;
    }

    p.lstm_param.input2input_weight       = REQUIRED_IO(inputs[3]);
    p.lstm_param.input2forget_weight      = REQUIRED_IO(inputs[4]);
    p.lstm_param.input2cell_weight        = REQUIRED_IO(inputs[5]);
    p.lstm_param.input2output_weight      = REQUIRED_IO(inputs[6]);
    p.lstm_param.recurrent2input_weight   = REQUIRED_IO(inputs[7]);
    p.lstm_param.recurrent2forget_weight  = REQUIRED_IO(inputs[8]);
    p.lstm_param.recurrent2cell_weight    = REQUIRED_IO(inputs[9]);
    p.lstm_param.recurrent2output_weight  = REQUIRED_IO(inputs[10]);
    p.lstm_param.input_gate_bias          = REQUIRED_IO(inputs[14]);
    p.lstm_param.forget_gate_bias         = REQUIRED_IO(inputs[15]);
    p.lstm_param.cell_bias                = REQUIRED_IO(inputs[16]);
    p.lstm_param.output_gate_bias         = OPTIONAL_IO(inputs[17]);
    p.lstm_param.projection_weight        = OPTIONAL_IO(inputs[18]);
    p.lstm_param.projection_bias          = OPTIONAL_IO(inputs[19]);

    p.lstm_param.activation = OPTIONAL_IO(self->nn_param.lstm.local.activation_tensor);
    p.lstm_param.cell_clip  = REQUIRED_IO(self->nn_param.lstm.local.cell_clip_tensor);
    p.lstm_param.proj_clip  = REQUIRED_IO(self->nn_param.lstm.local.proj_clip_tensor);

    self->n = vxLstmLayer(
        self->graph->g,
        REQUIRED_IO(inputs[0]),
        NULL,
        NULL,
        ( vx_nn_lstm_layer_params_t *)&p,
        sizeof( vx_nn_lstm_layer_params_t ),
        REQUIRED_IO(outputs[0])
        );
#endif

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
    if( inputs[0]->attr.dim_num  < 3)
    {
        VSILOGE( "Wrong shape parameters." );
        return FALSE;
    }
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = self->nn_param.lstm.weights;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 2];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        outputs[0]->attr.dim_num = 3;
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_tensor_t *activation_tensor, *forget_bias_tensor;
    vsi_nn_tensor_t *cell_clip_tensor, *proj_clip_tensor;
    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    activation_tensor = self->nn_param.lstm.local.activation_tensor;
    forget_bias_tensor = self->nn_param.lstm.local.forget_bias_tensor;
    cell_clip_tensor = self->nn_param.lstm.local.cell_clip_tensor;
    proj_clip_tensor = self->nn_param.lstm.local.proj_clip_tensor;
    if(NULL != self->n)
    {
        if(activation_tensor)vsi_nn_ReleaseTensor(&activation_tensor);
        if(forget_bias_tensor)vsi_nn_ReleaseTensor(&forget_bias_tensor);
        if(cell_clip_tensor)vsi_nn_ReleaseTensor(&cell_clip_tensor);
        if(proj_clip_tensor)vsi_nn_ReleaseTensor(&proj_clip_tensor);
        vxReleaseNode( &self->n );
        self->n = NULL;
    }

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTM,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 20,
    /* output_num */ 3
    );
#ifdef __cplusplus
}
#endif
