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
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

struct _scaletotensor_kernel_params
{
    int32_t ratio[2];
    int32_t offset[2];
    float mean[3];
    float scale;
};

typedef struct _scaletotensor_kernel_params scaletotensor_kernel_params_t;


static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;

    return status;
} /* op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_imageprocess_param * p;
    uint32_t i;
    p = (vsi_nn_imageprocess_param *)&(self->nn_param.imageprocess);
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        /* TODO */
        if (inputs[0]->attr.dim_num != 4)
        {
            VSILOGE("Only support 4D tensor for image process!(IMAGEPROCESS)\n");
            return FALSE;
        }
        if (p->reverse_channel == TRUE && inputs[0]->attr.size[2] != 3)
        {
            VSILOGE("Only support 3 channels for reverse channel!(IMAGEPROCESS)\n");
            return FALSE;
        }

        if (p->resize.type != VSI_NN_IMAGEPROCESS_RESIZE_NONE)
        {
            outputs[0]->attr.dim_num = p->resize.dim_num;
            for(i = 0; i < (uint32_t)p->resize.dim_num; ++i)
            {
                outputs[0]->attr.size[i] = p->resize.length[i];
            }
        }
        else if (p->crop.enable == TRUE)
        {
            outputs[0]->attr.dim_num = p->crop.dim_num;
            for(i = 0; i < (uint32_t)p->crop.dim_num; ++i)
            {
                outputs[0]->attr.size[i] = p->crop.length[i];
            }
        }
        else
        {
            // CWHN -> WHCN
            outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
            outputs[0]->attr.size[1] = inputs[0]->attr.size[2];
            outputs[0]->attr.size[2] = inputs[0]->attr.size[0];
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        }
    }
    return TRUE;
} /* op_setup() */

vsi_status vsi_nn_op_imageprocess_single_node
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_attr_t *attr,
    vsi_nn_imageprocess_param *p,
    uint8_t *data,
    vsi_nn_tensor_t *tensor_out
    )
{
    return VSI_SUCCESS;
}

vsi_status vsi_nn_ReleaseImageprocessSingleNode()
{
    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ IMAGEPROCESS,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ NULL,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
