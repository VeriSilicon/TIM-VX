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
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            ("com.vivantecorp.extension.Softmax2VXC")

#define SCALAR_INPUT_AXIS          (2)

__BEGIN_DECLS

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

DEF_KERNEL_INITIALIZER(_softmax_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t* param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    int sf_size = 0;
    vsi_nn_kernel_tensor_attr_t* attr = NULL;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        2,          // workdim
        {0, 0, 0},  // global_offset: control the start location be processed in the image
        {0, 0, 0},  // global_scale: how many pixels could be processed by a single thread
        {0, 0, 0},  // local_size: local group size in thread
        {0, 0, 0}}; // global_size: image size in thread

    VSI_UNREFERENCED(param_size);

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    if (!attr)
    {
        VSILOGE("Query failure! at line");
        return status;
    }

    sf_size  =  (int)attr->shape->data[0];

    gpu_param.global_offset[0] = 0;
    gpu_param.global_offset[1] = 0;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.local_size[0]    = 1;
    gpu_param.local_size[1]    = 1;
    gpu_param.global_size[0]   =
        gpu_align_p2((1 + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0],
                gpu_param.local_size[0]);
    gpu_param.global_size[1]   =
        gpu_align_p2((1 + gpu_param.global_scale[1] - 1) / gpu_param.global_scale[1],
                gpu_param.local_size[1]);
    {
        gpu_dp_inst_t Uni4x4_Fp16ToFp32 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node,
                "Uni4x4_Fp16ToFp32", &Uni4x4_Fp16ToFp32 );
        vsi_nn_kernel_gpu_add_param(node,
                "sf_size", &sf_size);
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

    if(status != VSI_SUCCESS)
    {
        VSILOGE("Initializer  failure!");
    }
    if (attr) vsi_nn_kernel_tensor_attr_release( &attr );

    return status;
}

static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    NULL,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    _softmax_initializer,
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

    vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
            "vsi_nn_kernel_header",
            "custom_softmax" );
    vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
            "custom_softmax" );
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

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            backend_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(
                    graph, I32, &axis );

            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &backend_params[SCALAR_INPUT_AXIS] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( custom_softmax, _setup )
