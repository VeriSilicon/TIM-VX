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
#include "utils/vsi_nn_util.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_internal_node.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    float width_ratio = self->nn_param.roi_align.width_ratio;
    float height_ratio = self->nn_param.roi_align.height_ratio;
    int32_t width_sample_num = self->nn_param.roi_align.width_sample_num;
    int32_t height_sample_num = self->nn_param.roi_align.height_sample_num;

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_float32( param, "width_ratio",  width_ratio );
    vsi_nn_kernel_param_add_float32( param, "height_ratio",  height_ratio );
    vsi_nn_kernel_param_add_int32( param, "width_sample_num",  width_sample_num );
    vsi_nn_kernel_param_add_int32( param, "height_sample_num",  height_sample_num );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "roi_align",
        inputs, 3,
        outputs, 1, param );

    if ( self->n )
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
    BEGIN_IO_TYPE_DECL(ROI_ALIGN, 3, 1)
        IO_TYPE(D_F16,       D_F16,         D_I32, D_F16)
        IO_TYPE(D_F16,       D_F16,         D_I32, D_F32)
        IO_TYPE(D_F16,       D_F32,         D_I32, D_F16)
        IO_TYPE(D_F32,       D_F32,         D_I32, D_F32)
        IO_TYPE(D_U8|Q_ASYM, D_U16|Q_ASYM,  D_I32, D_U8|Q_ASYM)
    END_IO_TYPE_DECL(ROI_ALIGN)
    if (!VALIDATE_OP_IO_TYPES(ROI_ALIGN, self, inputs, self->input.num, outputs, self->output.num))
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
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        vsi_nn_roi_align_param *p;
        p = &(self->nn_param.roi_align);
        outputs[0]->attr.dim_num = 4;
        outputs[0]->attr.size[0] = p->output_width;
        outputs[0]->attr.size[1] = p->output_height;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[1]->attr.size[1];
    }

    return TRUE;
} /* op_init() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ ROI_ALIGN,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
