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
#include <stdlib.h>
#include <math.h>
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vsi_nn_vxkernel.h"

#define _CPU_ARG_NUM            (1)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            ("com.vivantecorp.extension.CustomSampleVXC")

#define SCALAR_INPUT_AXIS          (3)

__BEGIN_DECLS

DEF_KERNEL_EXECUTOR(_softmax_compute)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t* param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    float *buffer[_CPU_IO_NUM] = {NULL};
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *attr[_CPU_IO_NUM] = {NULL};
    uint32_t i = 0, out_elements = 0;
    int32_t axis;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    tensors[0] = (vsi_nn_kernel_tensor_t)param[0]; // input0
    tensors[1] = (vsi_nn_kernel_tensor_t)param[1]; // input1
    tensors[2] = (vsi_nn_kernel_tensor_t)param[2]; // output

    attr[0] = vsi_nn_kernel_tensor_attr_create(tensors[0]);
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create(tensors[1]);
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create(tensors[2]);
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    /* alloc the float32 data buffer */
    buffer[0] = (float *)vsi_nn_kernel_tensor_create_buffer(tensors[0], attr[0], TRUE);
    CHECK_PTR_FAIL_GOTO(buffer[0], "Create input0 buffer fail.", final);

    buffer[1] = (float *)vsi_nn_kernel_tensor_create_buffer(tensors[1], attr[1], TRUE);
    CHECK_PTR_FAIL_GOTO(buffer[1], "Create input1 buffer fail.", final);

    out_elements = (uint32_t)vsi_nn_kernel_tensor_attr_get_size(attr[2]);
    buffer[2] = (float *)malloc(out_elements * sizeof(float));
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset(buffer[2], 0, out_elements * sizeof(float));

    /* CPU implement */
    for(i = 0; i < out_elements; i++)
    {
        buffer[2][i] = buffer[0][i] + buffer[1][0];
    }

    status = vsi_nn_kernel_tensor_write_from_float(
        tensors[2], attr[2], buffer[2], out_elements );
final:
    for(i = 0; i < _CPU_IO_NUM; i ++)
    {
        if(buffer[i])
        {
            free(buffer[i]);
        }
        vsi_nn_kernel_tensor_attr_release(&attr[i]);
    }
    return status;
}

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _softmax_compute,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
    return VSI_SUCCESS;
}

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
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    axis = vsi_nn_kernel_param_get_int32(params, "axis");
    status = _query_kernel(inputs, outputs, kernel);
    if(status != VSI_SUCCESS)
    {
        return NULL;
    }

    node = vsi_nn_kernel_create_node(graph, kernel);
    if(node == NULL)
    {
        return NULL;
    }

    /* Set inputs and outputs */
    vsi_nn_kernel_node_pack_io(backend_params, _CPU_PARAM_NUM,
            inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM);
    backend_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(
            graph, I32, &axis);

    /* Pass parameters to node. */
    status = vsi_nn_kernel_node_pass_param(node, backend_params, _CPU_PARAM_NUM);
    vsi_nn_kernel_scalar_release(&backend_params[SCALAR_INPUT_AXIS]);

    return node;
}

__END_DECLS

REGISTER_BACKEND_CPU( custom_sample, _setup )
