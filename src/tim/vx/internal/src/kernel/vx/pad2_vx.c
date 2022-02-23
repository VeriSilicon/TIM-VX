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
#include "utils/vsi_nn_dtype_util.h"

#define REGISTER_PAD2_OPENVX_KERNEL( kernel_name )   \
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

REGISTER_PAD2_OPENVX_KERNEL( pad2 )
{
    vx_node node = NULL;
    vx_nn_pad_params_t param;
    size_t dim_num = 0;
    int32_t* front_size = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "front_size", &dim_num);
    int32_t* back_size = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "back_size", &dim_num);
    int32_t pad_mode = vsi_nn_kernel_param_get_int32(params, "pad_mode");
    int32_t pad_front_array[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t pad_back_array[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_nn_tensor_t *convert_tensor = NULL;
    float const_val = vsi_nn_kernel_param_get_float32(params, "const_val");

    memset(&param, 0, sizeof(param));
    memset(pad_front_array, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    memset(pad_back_array, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    memcpy(pad_front_array, front_size, sizeof(int32_t) * dim_num);
    memcpy(pad_back_array, back_size, sizeof(int32_t) * dim_num);

    param.pad_mode = pad_mode;
    param.pad_const = vxCreateScalar( graph->ctx->c, VX_TYPE_FLOAT32, &const_val );
    param.numViewDimensions = (uint8_t)vsi_nn_max(dim_num, 2);
    param.pad_front_array = pad_front_array;
    param.pad_back_array = pad_back_array;

    if ( vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype) == FALSE)
    {
        vsi_nn_tensor_attr_t attr;
        memcpy( &attr, &outputs[0]->attr, sizeof( attr ) );
        memcpy( &attr.size, &inputs[0]->attr.size, sizeof( attr.size ) );
        attr.vtl = FALSE;
        attr.is_const = FALSE;

        convert_tensor = vsi_nn_CreateTensor(graph, &attr);

        node = vxTensorCopyNode(
            graph->g,
            inputs[0]->t,
            convert_tensor->t
            );
    }
    else
    {
        convert_tensor = vsi_nn_reshape_tensor( graph,
            inputs[0], inputs[0]->attr.size, inputs[0]->attr.dim_num );
    }

    node = vxTensorPadNode( graph->g, convert_tensor->t, outputs[0]->t, &param, sizeof(param) );

    vxReleaseScalar( &param.pad_const );
    vsi_safe_release_tensor(convert_tensor);

    return (vsi_nn_kernel_node_t)node;
} /* pad2() */

#undef REGISTER_PAD2_OPENVX_KERNEL
