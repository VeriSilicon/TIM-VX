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
#include "libnnext/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_util.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = vsi_nn_internal_compute_node( self );
    self->n = vsi_nn_internal_get_node_by_uid(self, 1)->node->n;
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_pre_process_param * p = NULL;
    vsi_bool ret = TRUE;
    vsi_nn_internal_tensor_t* preprocess_tensor = NULL;
    vsi_nn_preprocess_dest_layout_e layout = VSI_NN_DEST_LAYOUT_NCHW;

    p = (vsi_nn_pre_process_param *)&(self->nn_param.pre_process);

    vsi_nn_internal_init_node_wksp( self );

    if ( p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV420        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV444        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV12          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB           ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_BGRA          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR
        )
    {
        uint32_t i = 0;
        uint32_t _axis = 0;
        vsi_nn_tensor_attr_t attr;
        vsi_bool use_virtual_tensor = TRUE;


        for (i = 0; i < p->dim_num; i++)
        {
            _axis = p->perm[i];
            if (_axis != i)
                break;
        }

        if (i != self->nn_param.pre_process_rgb.dim_num)
        {
            layout = VSI_NN_DEST_LAYOUT_NHWC;
        }

        if (layout == VSI_NN_DEST_LAYOUT_NHWC)
        {
            memcpy( &attr, &outputs[PRE_PROCESS_OUTPUT]->attr, sizeof( attr ) );
            attr.size[0] = p->output_attr.size[1];
            attr.size[1] = p->output_attr.size[2];
            attr.size[2] = p->output_attr.size[0];
            p->output_attr.size[0] = (uint32_t)attr.size[0];
            p->output_attr.size[1] = (uint32_t)attr.size[1];
            p->output_attr.size[2] = (uint32_t)attr.size[2];
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;

            preprocess_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        }
    }

    switch (p->type)
    {
    case VSI_NN_SOURCE_FORMAT_TENSOR:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_TENSOR, 0, 0 );

            curr->node->nn_param.pre_process_tensor.perm = p->perm;
            curr->node->nn_param.pre_process_tensor.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_GRAY:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_GRAY, 0, 0 );

            curr->node->nn_param.pre_process_gray.mean = p->norm.mean[0];
            curr->node->nn_param.pre_process_gray.scale = p->norm.scale;
            curr->node->nn_param.pre_process_gray.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_gray.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_gray.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_gray.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_gray.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_gray.output_attr.dim_num = p->output_attr.dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_RGB:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_RGB, 0, 0 );

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[0];
            }
            else
            {
                curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[2];
            }

            curr->node->nn_param.pre_process_rgb.rgb_scale = p->norm.scale;
            curr->node->nn_param.pre_process_rgb.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_rgb.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_rgb.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_rgb.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_rgb.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_rgb.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_rgb.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_rgb.perm = p->perm;
            curr->node->nn_param.pre_process_rgb.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_YUV420:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_YUV420, 0, 0 );

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_yuv420.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv420.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv420.b_mean = p->norm.mean[0];
            }
            else
            {
                curr->node->nn_param.pre_process_yuv420.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv420.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv420.b_mean = p->norm.mean[2];
            }

            curr->node->nn_param.pre_process_yuv420.rgb_scale = p->norm.scale;
            curr->node->nn_param.pre_process_yuv420.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_yuv420.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_yuv420.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_yuv420.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_yuv420.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_yuv420.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_yuv420.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_yuv420.perm = p->perm;
            curr->node->nn_param.pre_process_yuv420.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            curr->inputs[1] = inputs[PRE_PROCESS_INPUT1];
            curr->inputs[2] = inputs[PRE_PROCESS_INPUT2];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_BGRA:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_BGRA, 0, 0 );

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_bgra.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_bgra.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_bgra.b_mean = p->norm.mean[0];
            }
            else
            {
                curr->node->nn_param.pre_process_bgra.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_bgra.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_bgra.b_mean = p->norm.mean[2];
            }

            curr->node->nn_param.pre_process_bgra.rgb_scale = p->norm.scale;
            curr->node->nn_param.pre_process_bgra.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_bgra.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_bgra.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_bgra.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_bgra.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_bgra.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_bgra.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_bgra.perm = p->perm;
            curr->node->nn_param.pre_process_bgra.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR:
        {
            uint32_t i = 0;
            uint32_t axis = 2;
            uint32_t group = 3;
            vsi_nn_tensor_t ** input_tensor_group = &p->local->local_tensor[0];
            vsi_nn_internal_tensor_t * output_tensor_group[3] = {NULL};
            vsi_nn_internal_tensor_t* tmp_outputs[3] = { NULL };
            vsi_nn_tensor_attr_t attr;
            float mean[3] = {0};
            vsi_size_t size_32bit[VSI_NN_MAX_DIM_NUM] = {0};

            memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

            ret = vsi_nn_CreateTensorGroup(self->graph, inputs[0], axis,
            input_tensor_group, group);
            if (ret == FALSE)
            {
                goto final;
            }

            memcpy(&attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t));
            for(i = 0; i < p->output_attr.dim_num; i++)
            {
                attr.size[i] = -1 == p->output_attr.size[i] ? -1 : (vsi_size_t)p->output_attr.size[i];
            }
            attr.size[axis] = 1;
            attr.vtl = TRUE;
            attr.is_const = FALSE;
            output_tensor_group[0] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            output_tensor_group[1] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            output_tensor_group[2] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
            {
                size_32bit[i] = attr.size[i];
            }

            if (p->reverse_channel)
            {
                int32_t order[3] = {2, 1, 0};

                mean[0] = p->norm.mean[2];
                mean[1] = p->norm.mean[1];
                mean[2] = p->norm.mean[0];

                vsi_nn_reorder_tensor( (vsi_nn_tensor_t **)output_tensor_group, order,
                        3, (vsi_nn_tensor_t **)tmp_outputs );
            }
            else
            {
                mean[0] = p->norm.mean[0];
                mean[1] = p->norm.mean[1];
                mean[2] = p->norm.mean[2];

                memmove( tmp_outputs, output_tensor_group, sizeof(vsi_nn_tensor_t*) * 3 );
            }

            for (i = 0; i < 3; i++)
            {
                curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_GRAY, 0, 0 );

                curr->node->nn_param.pre_process_gray.mean = mean[i];
                curr->node->nn_param.pre_process_gray.scale = p->norm.scale;
                curr->node->nn_param.pre_process_gray.rect.left = p->rect.left;
                curr->node->nn_param.pre_process_gray.rect.top = p->rect.top;
                curr->node->nn_param.pre_process_gray.rect.width = p->rect.width;
                curr->node->nn_param.pre_process_gray.rect.height = p->rect.height;
                curr->node->nn_param.pre_process_gray.output_attr.size = size_32bit;
                curr->node->nn_param.pre_process_gray.output_attr.dim_num = p->output_attr.dim_num;

                curr->inputs[0] = input_tensor_group[i];
                curr->outputs[0] = output_tensor_group[i]->t;

                vsi_nn_internal_setup_node(self, curr);
            }

            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, 3, 1 );

            curr->node->nn_param.concat.axis = axis;
            curr->inputs[0] = tmp_outputs[0]->t;
            curr->inputs[1] = tmp_outputs[1]->t;
            curr->inputs[2] = tmp_outputs[2]->t;
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_YUV444:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_YUV444, 0, 0 );

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_yuv444.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv444.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv444.b_mean = p->norm.mean[0];
            }
            else
            {
                curr->node->nn_param.pre_process_yuv444.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv444.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv444.b_mean = p->norm.mean[2];
            }

            curr->node->nn_param.pre_process_yuv444.rgb_scale = p->norm.scale;
            curr->node->nn_param.pre_process_yuv444.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_yuv444.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_yuv444.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_yuv444.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_yuv444.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_yuv444.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_yuv444.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_yuv444.perm = p->perm;
            curr->node->nn_param.pre_process_yuv444.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            curr->inputs[1] = inputs[PRE_PROCESS_INPUT1];
            curr->inputs[2] = inputs[PRE_PROCESS_INPUT2];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_NV12:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_NV12, 0, 0 );

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_nv12.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_nv12.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_nv12.b_mean = p->norm.mean[0];
            }
            else
            {
                curr->node->nn_param.pre_process_nv12.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_nv12.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_nv12.b_mean = p->norm.mean[2];
            }

            curr->node->nn_param.pre_process_nv12.rgb_scale = p->norm.scale;
            curr->node->nn_param.pre_process_nv12.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_nv12.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_nv12.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_nv12.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_nv12.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_nv12.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_nv12.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_nv12.perm = p->perm;
            curr->node->nn_param.pre_process_nv12.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            curr->inputs[1] = inputs[PRE_PROCESS_INPUT1];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            vsi_nn_internal_setup_node(self, curr);
        }
        break;
    default:
        {
            VSILOGE( "Not support this type!(PRE_PROCESS)\n");
            ret = FALSE;
        }
        break;
    }

    if ( p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV420        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV444        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV12          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB           ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_BGRA          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR
        )
    {
        if (layout == VSI_NN_DEST_LAYOUT_NHWC)
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
            curr->node->nn_param.permute.perm = p->perm;
            curr->node->nn_param.permute.dim_num = p->dim_num;
            curr->inputs[0] = preprocess_tensor->t;
            curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

            vsi_nn_internal_setup_node( self, curr );
        }
    }

final:

    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );

    if (self->nn_param.pre_process.local != NULL)
    {
        uint32_t i = 0;

        for ( i = 0; i < _VSI_NN_PRE_PROCESS_LOCAL_TENSOR_NUM; i++)
        {
            if (self->nn_param.pre_process.local->local_tensor[i] != NULL)
            {
                vsi_nn_ReleaseTensor(&(self->nn_param.pre_process.local->local_tensor[i]));
            }
        }

        free(self->nn_param.pre_process.local);
        self->nn_param.pre_process.local = NULL;
    }

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.pre_process.local   =
    (vsi_nn_pre_process_lcl_data *)malloc(sizeof(vsi_nn_pre_process_lcl_data));

    if (NULL == self->nn_param.pre_process.local)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(self->nn_param.pre_process.local, 0, sizeof(vsi_nn_pre_process_lcl_data));

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PRE_PROCESS,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ PRE_PROCESS_INPUT_CNT,
    /* output_num */ PRE_PROCESS_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
