/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#define REGISTER_SOFTMAX_OPENVX_KERNEL( kernel_name )   \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ); \
    REGISTER_BACKEND_OPENVX( kernel_name, _##kernel_name##setup ) \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        )

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vx_node node = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );
    int32_t type  = vsi_nn_kernel_param_get_int32( params, "type" );

#ifdef VX_SCALE_EXTRA_PARAMETER_SUPPORT
    vx_nn_scale_params_ext_t param;
    param.align_corners = align_corners;
    param.half_pixel_centers = half_pixel_centers;
    switch (type)
    {
        case VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR:
            param.base.type = VX_INTERPOLATION_NEAREST_NEIGHBOR;
            break;
        case VSI_NN_INTERPOLATION_BILINEAR:
            param.base.type = VX_INTERPOLATION_BILINEAR;
            break;
        case VSI_NN_INTERPOLATION_AREA:
            param.base.type = VX_INTERPOLATION_AREA;
            break;
        default:
            param.base.type = VX_INTERPOLATION_NEAREST_NEIGHBOR;
    }
    node = vxTensorScaleNode( graph->g,
                              inputs[0]->t,
                              (vx_nn_scale_params)(&param),
                              sizeof(vx_nn_scale_params_ext_t),
                              outputs[0]->t );
#else
    vx_nn_scale_params_t param;
    if (align_corners || half_pixel_centers)
    {
        return NULL;
    }
    switch (type)
    {
        case VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR:
            param.type = VX_INTERPOLATION_NEAREST_NEIGHBOR;
            break;
        case VSI_NN_INTERPOLATION_BILINEAR:
            param.type = VX_INTERPOLATION_BILINEAR;
            break;
        case VSI_NN_INTERPOLATION_AREA:
            param.type = VX_INTERPOLATION_AREA;
            break;
        default:
            param.type = VX_INTERPOLATION_NEAREST_NEIGHBOR;
            break;
    }

    node = vxTensorScaleNode( graph->g,
                              inputs[0]->t,
                              &param,
                              sizeof(param),
                              outputs[0]->t );
#endif
    if ( NULL == node )
    {
        VSILOGI("Call vxTensorScaleNode fail.(resize)");
    }

    return (vsi_nn_kernel_node_t)node;
} /* _setup() */

#define REGISTER_RESIZE_OPENVX_KERNEL(KERNEL_NAME) \
    static vsi_nn_kernel_node_t _##KERNEL_NAME##_setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num, \
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ) \
    { \
        return _setup(graph, inputs, input_num, outputs, output_num, \
                params, kernel); \
    } \
    REGISTER_BACKEND_OPENVX( KERNEL_NAME, _##KERNEL_NAME##_setup )

REGISTER_RESIZE_OPENVX_KERNEL( resize_nearest )
REGISTER_RESIZE_OPENVX_KERNEL( resize_bilinear )

#undef REGISTER_RESIZE_OPENVX_KERNEL
