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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "space2depth_internal"

#define HASH_SPACE2DEPTH_INTERNAL_KEY(_input0_type, _output_type, _opt_flg) \
    ((_input0_type << 24) | (_output_type << 16) | (_opt_flg << 8))

#define HASH_SPACE2DEPTH_INTERNAL_CL_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.space2depth_internal_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_SPACE2DEPTH_INTERNAL_X2Y1_CL_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.space2depth_internal_"#SRC0_TYPE"to"#DST_TYPE"_X2Y1")

#define TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SPACE2DEPTH_INTERNAL_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_SPACE2DEPTH_INTERNAL_CL_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

 #define TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SPACE2DEPTH_INTERNAL_KEY(IN0_TYPE, OUT_TYPE, 1), \
        HASH_SPACE2DEPTH_INTERNAL_X2Y1_CL_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(F32, F32,  KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(U8, U8, KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(F32, F32,  KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(U8, U8, KERNEL_SOURCE_1)
};

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_space2depth_internal_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_status    status             = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * in_shape = NULL;
    vsi_ssize_t width = 0;
    vsi_ssize_t height = 0;
    vsi_ssize_t chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    in_shape  = attr[0]->shape;
    width = in_shape->data[0];
    height = in_shape->data[1];
    chn = in_shape->size > 2 ? in_shape->data[2] : 1;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = chn;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }

    return status;
} /* _space2depth_internal_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t opt_flg
    )
{
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_SPACE2DEPTH_INTERNAL_KEY( input0_dtype, output_dtype, opt_flg );

    if (input0_dtype == F16 && output_dtype == F16)
    {
        input0_dtype = F32;
        output_dtype = F32;
    }

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _space2depth_internal_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t block_size_x  = vsi_nn_kernel_param_get_int32( params, "block_size_x" );
    int32_t block_size_y  = vsi_nn_kernel_param_get_int32( params, "block_size_y" );
    int32_t opt_flg = (block_size_x == 2 && block_size_y == 1) ? 1 : 0;

    float inputScale = inputs[0]->attr.dtype.scale;
    int32_t inputZp = inputs[0]->attr.dtype.zero_point;
    float outputScale = outputs[0]->attr.dtype.scale;
    int32_t outputZp = outputs[0]->attr.dtype.zero_point;
    float scaleInOut = 1.0f;
    float zpInOut = 0.0f;

    if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        int32_t input_fl = inputs[0]->attr.dtype.fl;
        if (input_fl > 0)
        {
            inputScale = (1.0f / ((float) ((int64_t)1 << input_fl)));
        }
        else
        {
            inputScale = ((float) ((int64_t)1 << -input_fl));
        }
        inputZp = 0;
    }
    else if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_NONE)
    {
        inputScale = 1.0f;
        inputZp = 0;
    }

    if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        int32_t output_fl = outputs[0]->attr.dtype.fl;
        if (output_fl > 0)
        {
            outputScale = (1.0f / ((float) ((int64_t)1 << output_fl)));
        }
        else
        {
            outputScale = ((float) ((int64_t)1 << -output_fl));
        }
        outputZp = 0;
    }
    else if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_NONE)
    {
        outputScale = 1.0f;
        outputZp = 0;
    }
    scaleInOut = inputScale / outputScale;
    zpInOut = outputZp - inputZp * scaleInOut;

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, opt_flg);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if ( node )
        {
            int32_t index = 2;
            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    inputs, 1, outputs, 1 );
            node_params[index++] = vsi_nn_kernel_scalar_create(
                    graph, I32, &block_size_x );
            node_params[index++] = vsi_nn_kernel_scalar_create(
                    graph, I32, &block_size_y );
            node_params[index++] = vsi_nn_kernel_scalar_create(
                    graph, F32, &scaleInOut );
            node_params[index] = vsi_nn_kernel_scalar_create(
                    graph, F32, &zpInOut );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( space2depth_internal, _setup )

