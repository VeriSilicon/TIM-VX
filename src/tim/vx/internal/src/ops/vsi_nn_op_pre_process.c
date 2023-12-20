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
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_error.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* interal_node = NULL;

    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);

    status = vsi_nn_internal_compute_node( self );
    CHECK_STATUS_FAIL_GOTO(status, final );

    interal_node = vsi_nn_internal_get_node_by_uid(self, 1);

    if (interal_node)
    {
        self->n = interal_node->node->n;
    }
    else
    {
        status = VSI_FAILURE;
    }

final:
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
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
    vsi_bool ret = FALSE;
    vsi_nn_internal_tensor_t* preprocess_tensor = NULL;
    vsi_nn_preprocess_dest_layout_e layout = VSI_NN_DEST_LAYOUT_NCHW;
    vsi_bool enable_rgb88_planar_nhwc = FALSE;

    p = (vsi_nn_pre_process_param *)&(self->nn_param.pre_process);

    vsi_nn_internal_init_node_wksp( self );

    if (vsi_nn_compareVersion(self->graph, 1, 1, 83) == -1)
    {
        p->norm2.scale[0] = p->norm.scale;
        p->norm2.scale[1] = p->norm.scale;
        p->norm2.scale[2] = p->norm.scale;
    }

    if ( p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV420        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV444        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV12          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV21          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB           ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_BGRA          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_GRAY          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR_SEP ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUYV422       ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_UYVY422
        )
    {
        uint32_t i = 0;
        uint32_t _axis = 0;
        vsi_nn_tensor_attr_t attr;
        vsi_bool use_virtual_tensor = TRUE;

        for (i = 0; i < p->dim_num; i++)
        {
            if (p->perm == NULL)
            {
                i = p->dim_num;
                break;
            }
            _axis = p->perm[i];
            if (_axis != i)
                break;
        }

        if (i != self->nn_param.pre_process_rgb.dim_num)
        {
            layout = VSI_NN_DEST_LAYOUT_NHWC;

            if (p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR ||
                p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR_SEP)
            {
                enable_rgb88_planar_nhwc = self->graph->ctx->options.enable_rgb88_planar_nhwc;
            }
        }

        if (layout == VSI_NN_DEST_LAYOUT_NHWC && !enable_rgb88_planar_nhwc)
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

            preprocess_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
            CHECK_PTR_FAIL_GOTO(preprocess_tensor, "Create internal tensor failed", final);
        }
    }

    switch (p->type)
    {
    case VSI_NN_SOURCE_FORMAT_TENSOR:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_TENSOR, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            curr->node->nn_param.pre_process_tensor.perm = p->perm;
            curr->node->nn_param.pre_process_tensor.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_GRAY:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_GRAY, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            curr->node->nn_param.pre_process_gray.mean = p->norm.mean[0];
            curr->node->nn_param.pre_process_gray.scale = p->norm2.scale[0];
            curr->node->nn_param.pre_process_gray.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_gray.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_gray.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_gray.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_gray.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_gray.output_attr.dim_num = p->output_attr.dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_RGB:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_RGB, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_rgb.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_rgb.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_rgb.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_rgb.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_rgb.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_rgb.b_scale = p->norm2.scale[2];
            }

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

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_YUV420:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_YUV420, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_yuv420.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv420.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv420.b_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv420.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_yuv420.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_yuv420.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_yuv420.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv420.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv420.b_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv420.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_yuv420.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_yuv420.b_scale = p->norm2.scale[2];
            }

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

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_BGRA:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_BGRA, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_bgra.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_bgra.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_bgra.b_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_bgra.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_bgra.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_bgra.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_bgra.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_bgra.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_bgra.b_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_bgra.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_bgra.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_bgra.b_scale = p->norm2.scale[2];
            }

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

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR:
    case VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR_SEP:
        {
            vsi_bool is_input_sep = p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR ? FALSE : TRUE;
            float mean[3] = {0};

            if (p->reverse_channel)
            {
                mean[0] = p->norm.mean[2];
                mean[1] = p->norm.mean[1];
                mean[2] = p->norm.mean[0];
            }
            else
            {
                mean[0] = p->norm.mean[0];
                mean[1] = p->norm.mean[1];
                mean[2] = p->norm.mean[2];
            }

            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_RGB888_PLANAR, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
            if (is_input_sep)
            {
                curr->inputs[0] = inputs[0];
                curr->inputs[1] = inputs[1];
                curr->inputs[2] = inputs[2];
            }
            else
            {
                curr->inputs[0] = inputs[0];
                curr->inputs[1] = NULL;
                curr->inputs[2] = NULL;
            }
            if (layout == VSI_NN_DEST_LAYOUT_NHWC && !enable_rgb88_planar_nhwc)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_rgb888_planar.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_rgb888_planar.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_rgb888_planar.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_rgb888_planar.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_rgb888_planar.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_rgb888_planar.b_scale = p->norm2.scale[2];
            }

            curr->node->nn_param.pre_process_rgb888_planar.r_mean = mean[0];
            curr->node->nn_param.pre_process_rgb888_planar.g_mean = mean[1];
            curr->node->nn_param.pre_process_rgb888_planar.b_mean = mean[2];
            curr->node->nn_param.pre_process_rgb888_planar.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_rgb888_planar.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_rgb888_planar.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_rgb888_planar.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_rgb888_planar.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_rgb888_planar.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_rgb888_planar.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_rgb888_planar.enable_rgb88_planar_nhwc = enable_rgb88_planar_nhwc;
            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_YUV444:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_YUV444, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_yuv444.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv444.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv444.b_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv444.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_yuv444.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_yuv444.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_yuv444.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv444.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv444.b_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv444.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_yuv444.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_yuv444.b_scale = p->norm2.scale[2];
            }

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

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_NV21:
    case VSI_NN_SOURCE_FORMAT_IMAGE_NV12:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_NV12, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_nv12.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_nv12.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_nv12.b_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_nv12.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_nv12.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_nv12.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_nv12.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_nv12.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_nv12.b_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_nv12.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_nv12.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_nv12.b_scale = p->norm2.scale[2];
            }

            if (p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV12)
            {
                curr->node->nn_param.pre_process_nv12.nv_type = VSI_NN_YUV_TYPE_NV12;
            }
            else
            {
                curr->node->nn_param.pre_process_nv12.nv_type = VSI_NN_YUV_TYPE_NV21;
            }

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

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    case VSI_NN_SOURCE_FORMAT_IMAGE_YUYV422:
    case VSI_NN_SOURCE_FORMAT_IMAGE_UYVY422:
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PRE_PROCESS_YUV422, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);

            if (p->reverse_channel)
            {
                curr->node->nn_param.pre_process_yuv422.r_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv422.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv422.b_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv422.r_scale = p->norm2.scale[2];
                curr->node->nn_param.pre_process_yuv422.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_yuv422.b_scale = p->norm2.scale[0];
            }
            else
            {
                curr->node->nn_param.pre_process_yuv422.r_mean = p->norm.mean[0];
                curr->node->nn_param.pre_process_yuv422.g_mean = p->norm.mean[1];
                curr->node->nn_param.pre_process_yuv422.b_mean = p->norm.mean[2];
                curr->node->nn_param.pre_process_yuv422.r_scale = p->norm2.scale[0];
                curr->node->nn_param.pre_process_yuv422.g_scale = p->norm2.scale[1];
                curr->node->nn_param.pre_process_yuv422.b_scale = p->norm2.scale[2];
            }

            if (p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUYV422)
            {
                curr->node->nn_param.pre_process_yuv422.yuv422_type = 0;
            }
            else
            {
                curr->node->nn_param.pre_process_yuv422.yuv422_type = 1;
            }

            curr->node->nn_param.pre_process_yuv422.reverse_channel = p->reverse_channel;
            curr->node->nn_param.pre_process_yuv422.rect.left = p->rect.left;
            curr->node->nn_param.pre_process_yuv422.rect.top = p->rect.top;
            curr->node->nn_param.pre_process_yuv422.rect.width = p->rect.width;
            curr->node->nn_param.pre_process_yuv422.rect.height = p->rect.height;
            curr->node->nn_param.pre_process_yuv422.output_attr.size = p->output_attr.size;
            curr->node->nn_param.pre_process_yuv422.output_attr.dim_num = p->output_attr.dim_num;
            curr->node->nn_param.pre_process_yuv422.perm = p->perm;
            curr->node->nn_param.pre_process_yuv422.dim_num = p->dim_num;

            curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
            if (layout == VSI_NN_DEST_LAYOUT_NHWC)
            {
                curr->outputs[0] = preprocess_tensor->t;
            }
            else
            {
                curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];
            }

            ret = vsi_nn_internal_setup_node(self, curr);
        }
        break;
    default:
        {
            VSILOGE( "Not support this type!(PRE_PROCESS)\n");
            goto final;
        }
        break;
    }

    if ( p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV420        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUYV422       ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_UYVY422       ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_YUV444        ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV12          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_NV21          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB           ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_BGRA          ||
         p->type == VSI_NN_SOURCE_FORMAT_IMAGE_GRAY          ||
         (p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR && !enable_rgb88_planar_nhwc) ||
         (p->type == VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR_SEP && !enable_rgb88_planar_nhwc)
        )
    {
        if (layout == VSI_NN_DEST_LAYOUT_NHWC)
        {
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
            CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
            curr->node->nn_param.permute.perm = p->perm;
            curr->node->nn_param.permute.dim_num = p->dim_num;
            curr->inputs[0] = preprocess_tensor->t;
            curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

            ret = vsi_nn_internal_setup_node( self, curr );
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
