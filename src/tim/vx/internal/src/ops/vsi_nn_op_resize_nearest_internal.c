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
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    int32_t  align_corners       = self->nn_param.resize_nearest_internal.align_corners;
    int32_t  half_pixel_centers  = self->nn_param.resize_nearest_internal.half_pixel_centers;
    vsi_nn_kernel_param_t * param = NULL;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "align_corners",  align_corners );
    vsi_nn_kernel_param_add_int32( param, "half_pixel_centers",  half_pixel_centers );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
            "resize_nearest",
            &inputs[0], 1,
            &outputs[0], 1, param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;

} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(RESIZE_NEAREST_INTERNAL, 1, 1)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
    END_IO_TYPE_DECL(RESIZE_NEAREST_INTERNAL)
    if(!VALIDATE_OP_IO_TYPES(RESIZE_NEAREST_INTERNAL, self, inputs, self->input.num, outputs, self->output.num)) {
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
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    float factor = self->nn_param.resize_nearest_internal.factor;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        if (factor != 0)
        {
            outputs[0]->attr.size[0] = (uint32_t)(inputs[0]->attr.size[0] * factor);
            outputs[0]->attr.size[1] = (uint32_t)(inputs[0]->attr.size[1] * factor);
        }
        else
        {
            outputs[0]->attr.size[0] = self->nn_param.resize.size[0];
            outputs[0]->attr.size[1] = self->nn_param.resize.size[1];
        }
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.resize_nearest_internal.lcl_data_ptr)
    {
        free(self->nn_param.resize_nearest_internal.lcl_data_ptr);
        self->nn_param.resize_nearest_internal.lcl_data_ptr = NULL;
    }

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */


static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.resize_nearest_internal.lcl_data_ptr   =
    (vsi_nn_resize_nearest_in_lcl_data *)malloc(sizeof(vsi_nn_resize_nearest_in_lcl_data));
    if (NULL == self->nn_param.resize_nearest_internal.lcl_data_ptr)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset(self->nn_param.resize_nearest_internal.lcl_data_ptr, 0, sizeof(vsi_nn_resize_nearest_in_lcl_data));

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RESIZE_NEAREST_INTERNAL,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
