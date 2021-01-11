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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_nn_hashlut_params_t p;
    memset( &p, 0, sizeof(p) );
    p.keys = inputs[1]->t;
    p.values = inputs[2]->t;
    self->n = vxHashTableLookupLayer( self->graph->g, inputs[0]->t,
            &p, sizeof(p), outputs[1]->t, outputs[0]->t);
    if( !self->n )
    {
        status = VSI_FAILURE;
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
    BEGIN_IO_TYPE_DECL(HASHTABLE_LOOKUP, 3, 2)
        IO_TYPE(D_I32, D_I32,  D_F16, D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_I32, D_I32,  D_I32, D_U8|Q_ASYM, D_I32)
        IO_TYPE(D_I32, D_I32,  D_F32, D_U8|Q_ASYM, D_F32)
        IO_TYPE(D_I32, D_I32,  D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_I32, D_I32,  D_F16, D_F16, D_F16)
        IO_TYPE(D_I32, D_I32,  D_F16, D_F16, D_U8|Q_ASYM)
        IO_TYPE(D_I32, D_I32,  D_I32, D_F16, D_I32)
        IO_TYPE(D_I32, D_I32,  D_F32, D_F16, D_F32)
        IO_TYPE(D_I32, D_I32,  D_U8|Q_ASYM, D_F16, D_U8|Q_ASYM)
    END_IO_TYPE_DECL(HASHTABLE_LOOKUP)
    if (!VALIDATE_OP_IO_TYPES(HASHTABLE_LOOKUP, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if( outputs[0]->attr.dim_num == VSI_NN_DIM_AUTO )
    {
        outputs[0]->attr.dim_num = inputs[2]->attr.dim_num;
        memcpy( outputs[0]->attr.size, inputs[2]->attr.size,
                sizeof(int) * inputs[2]->attr.dim_num );
        outputs[0]->attr.size[outputs[0]->attr.dim_num - 1] = inputs[0]->attr.size[0];
    }
    if( outputs[1]->attr.dim_num == VSI_NN_DIM_AUTO )
    {
        outputs[1]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[1]->attr.size[0] = inputs[0]->attr.size[0];
    }
    return TRUE;
} /* op_setup() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ HASHTABLE_LOOKUP,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 2
    );
#ifdef __cpluplus
}
#endif
