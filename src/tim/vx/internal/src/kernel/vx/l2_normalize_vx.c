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

#define REGISTER_L2_NORMALIZE_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_L2_NORMALIZE_OPENVX_KERNEL( l2_norm )
{
    vx_node node = NULL;
    int32_t axis = vsi_nn_kernel_param_get_int32(params, "axis");
#ifdef VX_L2NORM_AXIS_PARAMETER_SUPPORT
    vx_nn_l2norm_params_t param;

    param.axis = axis;

    node = vxL2NormalizeLayer2(
        graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_l2norm_params_t),
        outputs[0]->t
        );
#else
    uint32_t i = 0;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t innerSize = 1;
    uint32_t outerSize = 1;
    uint32_t axisSize  = 1;
    vx_tensor vx_input = NULL;
    vx_tensor vx_output = NULL;
    vx_tensor input = inputs[0]->t;
    vx_tensor output = outputs[0]->t;

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

    node = vxL2NormalizeLayer(
        graph->g,
        input,
        output
        );

    if (vx_input) vxReleaseTensor(&vx_input);
    if (vx_output) vxReleaseTensor(&vx_output);
#endif

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);

    if( NULL == node )
    {
        VSILOGE("Call vxSoftmaxLayer2 fail.(softmax)");
    }

    return (vsi_nn_kernel_node_t)node;
} /* l2_norm() */

#undef REGISTER_L2_NORMALIZE_OPENVX_KERNEL
