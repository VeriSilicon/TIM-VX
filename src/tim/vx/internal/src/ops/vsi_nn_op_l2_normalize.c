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
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VX_FAILURE;
#ifdef VX_L2NORM_AXIS_PARAMETER_SUPPORT
    vx_nn_l2norm_params_t param;

    param.axis = self->nn_param.l2_normalize.axis;

    self->n = vxL2NormalizeLayer2(
        self->graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_l2norm_params_t),
        outputs[0]->t
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
#else
    vsi_nn_l2_normalize_param * p;
    int32_t axis = -1;
    uint32_t i = 0;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t innerSize = 1;
    uint32_t outerSize = 1;
    uint32_t axisSize  = 1;
    vx_tensor vx_input = NULL;
    vx_tensor vx_output = NULL;
    vx_tensor input = inputs[0]->t;
    vx_tensor output = outputs[0]->t;

    status = VSI_FAILURE;

    p = &(self->nn_param.l2_normalize);
    axis = p->axis;

    if (axis != 2)
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

    self->n = vxL2NormalizeLayer(
        self->graph->g,
        input,
        output
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    if (vx_input) vxReleaseTensor(&vx_input);
    if (vx_output) vxReleaseTensor(&vx_output);
#endif
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
        self->nn_param.l2_normalize.axis = 2;
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
    BEGIN_IO_TYPE_DECL(L2_NORMALIZE, 1, 1)
        IO_TYPE(D_F32,  D_F16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_F16,  D_F32)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
    END_IO_TYPE_DECL(L2_NORMALIZE)
    if (!VALIDATE_OP_IO_TYPES(L2_NORMALIZE, self, inputs, self->input.num, outputs, self->output.num))
    {
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
    /* op_name    */ L2_NORMALIZE,
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

