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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_constraint_check.h"

#ifdef __cplusplus
extern "C" {
#endif

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_tensor_t* act_tensor = NULL;
    vx_nn_rnn_params_t param;

    memset(&param, 0, sizeof(vx_nn_rnn_params_t));

    act_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t*)&self->nn_param.rnn.activation,
        VSI_NN_TYPE_INT32);

    if (!act_tensor)
    {
        VSILOGE("RNN->Create Activation Tensor failed");
        status = VSI_FAILURE;
    }
    else
    {
        param.weights            = REQUIRED_IO(inputs[1]);
        param.recurrent_weights  = REQUIRED_IO(inputs[2]);
        param.bias               = REQUIRED_IO(inputs[3]);
        param.state_in           = REQUIRED_IO(inputs[4]);
        param.activation         = REQUIRED_IO(act_tensor);
        self->n = vxRNNLayer(
                    self->graph->g,
                    REQUIRED_IO(inputs[0]),
                    &param,
                    sizeof(param),
                    /*state output*/REQUIRED_IO(outputs[0]),
                    /*output*/REQUIRED_IO(outputs[1]));

        vsi_nn_ReleaseTensor(&act_tensor);
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
    uint32_t input_idx = 0;
    do {
        vsi_bool break_early = FALSE;

        // input_idx = 0 : inputs[0].shape = shape(batch_size, input_size)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        // input_idx = 1 : inputs[1].shape = shape(num_units, input_size)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        // input_idx = 2 : inputs[2].shape = shape(num_units, num_units)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        // input_idx = 3 : inputs[3].shape = shape(num_units)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 1);
        if (break_early) break;
        input_idx ++;

        // input_idx = 4 : inputs[4].shape = shape(batch_size, num_units)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        return TRUE;
    } while(0);

    {
        BEGIN_IO_TYPE_DECL(RNN, 5, 1)
            IO_TYPE(D_F16,  D_F16, D_F16, D_F16, D_F16, D_F16)
            IO_TYPE(D_F16,  D_F16, D_F16, D_F32, D_F16, D_F16)
            IO_TYPE(D_F32,  D_F32, D_F32, D_F32, D_F32, D_F32)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
        END_IO_TYPE_DECL(RNN)
        if(!VALIDATE_OP_IO_TYPES(RNN, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    VSILOGE("RNN check shape faild at Input[%d]", input_idx);



    return FALSE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num) {
        outputs[0]->attr.size[0] = inputs[4]->attr.size[0];
        outputs[0]->attr.size[1] = inputs[4]->attr.size[1];
        outputs[1]->attr.size[0] = inputs[4]->attr.size[0];
        outputs[1]->attr.size[1] = inputs[4]->attr.size[1];

        outputs[0]->attr.dim_num = outputs[1]->attr.dim_num = inputs[4]->attr.dim_num;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (NULL == self)
    {
        return VSI_FAILURE;
    }

    if (NULL != self->n)
    {
        vxReleaseNode(&self->n);
        self->n = NULL;
    }

    return VSI_SUCCESS;
} /* op_deinit() */

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RNN,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 5,
    /* output_num */ 2
    );
#ifdef __cplusplus
}
#endif
