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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    int32_t input_size[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dims = 0;
    vx_tensor input = NULL, input0 = NULL;
    vx_tensor output = NULL, output0 = NULL;
    status = VSI_FAILURE;

    if (inputs[0]->attr.dim_num > 4)
    {
        input_size[0] = vsi_nn_GetElementNum(inputs[0]) /
            inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        input_size[1] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        dims= 2;
        input = vxReshapeTensor(inputs[0]->t, input_size, dims);
        output = vxReshapeTensor(outputs[0]->t, input_size, dims);
        input0 = input;
        output0 = output;
    }
    else
    {
        input0 = inputs[0]->t;
        output0 = outputs[0]->t;
    }

    self->n = vxLeakyReluLayer(
        self->graph->g,
        input0,
        -1,
        output0
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    if (input)  vxReleaseTensor(&input);
    if (output) vxReleaseTensor(&output);
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* check inputs outputs data type */
    BEGIN_IO_TYPE_DECL(ABS, 1, 1)
        /* IO_TYPE(INPUT, OUTPUT) */
        IO_TYPE(D_F32, D_F32)
        IO_TYPE(D_F32, D_F16)
        IO_TYPE(D_F32, D_BF16)

        IO_TYPE(D_F16, D_F32)
        IO_TYPE(D_F16, D_F16)
        IO_TYPE(D_F16, D_U8|Q_ASYM)
        IO_TYPE(D_F16, D_I8|Q_DFP)
        IO_TYPE(D_F16, D_I16|Q_DFP)

        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_BF16, D_F32)

        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_F16)

        IO_TYPE(D_I8|Q_ASYM, D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM, D_F16)

        IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP, D_F16)

        IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP, D_F16)
    END_IO_TYPE_DECL(ABS)
    if(!VALIDATE_OP_IO_TYPES(ABS, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ ABS,
    /* init       */ NULL,
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

