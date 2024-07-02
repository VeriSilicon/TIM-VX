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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_error.h"

typedef struct _bitcast_local_data_t {
    int32_t placeholder;
} bitcast_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status           status  = VSI_FAILURE;
    vsi_nn_kernel_node_t n       = NULL;

    n = vsi_nn_kernel_selector( self->graph, "bitcast", inputs, 1, outputs, 1, NULL );
    if (n != NULL)
    {
        status = VSI_SUCCESS;
    }
    self->n = (vx_node)n;

    return status;
} /* op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t i = 0;

    VSI_UNREFERENCED(self);

    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        uint32_t input_byte = 0;
        uint32_t output_byte = 0;
        uint32_t in_dim = inputs[0]->attr.dim_num;
        input_byte = vsi_nn_TypeGetBytesExt(inputs[0]->attr.dtype.vx_type);
        output_byte = vsi_nn_TypeGetBytesExt(outputs[0]->attr.dtype.vx_type);

        if (input_byte == output_byte)
        {
            outputs[0]->attr.dim_num = in_dim;
            for (i = 0; i < (int32_t)(in_dim); i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
            }
        }
        else if (input_byte > output_byte)
        {
            outputs[0]->attr.dim_num = in_dim + 1;
            outputs[0]->attr.size[0] = input_byte / output_byte;
            for (i = 1;i < (int32_t)(outputs[0]->attr.dim_num); i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i - 1];
            }
        }
        else
        {
            if ((uint32_t)(inputs[0]->attr.size[in_dim - 1]) != output_byte / input_byte)
            {
                VSILOGE("If input datatype is smaller than output datatype, bitcast op requires that \
                    the rightmost dimension be equal to sizeof(output datatype) / sizeof(input datatype)");
                return FALSE;
            }
            outputs[0]->attr.dim_num = in_dim - 1;
            if (outputs[0]->attr.dim_num == 0)
            {
                outputs[0]->attr.size[0] = 1;
                vsi_nn_SetTensorIsScalar(outputs[0], TRUE);
            }
            else
            {
                for (i = 0; i < (int32_t)(outputs[0]->attr.dim_num); i++)
                {
                    outputs[0]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
            }
        }
    }

    return TRUE;
} /* op_setup() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BITCAST,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ NULL,
    /* check      */ NULL,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

