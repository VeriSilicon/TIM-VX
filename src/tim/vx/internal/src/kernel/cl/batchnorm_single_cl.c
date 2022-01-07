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

__BEGIN_DECLS


/*
 * Define kernel meta.
 */
#define HASH_BATCH_NORM_KEY( _input_type, _output_type, _image_2d) \
    ( (_input_type << 12) | (_output_type << 4) | (_image_2d))

 #define VSI_NN_GEN_BATCH_NORM_KERNEL_SOURCE_NAME \
    "batchnorm_single"

#define HASH_BATCH_NORM_SH_KERNEL_NAME( SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.batch_norm_"#SRC_TYPE"to"#DST_TYPE)

#define TENSOR_BATCH_NORM_KERNELS( SRC_TYPE, OUT_TYPE) \
    {   HASH_BATCH_NORM_KEY( SRC_TYPE, OUT_TYPE, 0), \
        HASH_BATCH_NORM_SH_KERNEL_NAME( SRC_TYPE, OUT_TYPE), \
        VSI_NN_GEN_BATCH_NORM_KERNEL_SOURCE_NAME },

#define HASH_BATCH_NORM_SH_KERNEL_2D_NAME( SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("batch_norm_"#SRC_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_BATCH_NORM_KERNELS_2D( SRC_TYPE, OUT_TYPE) \
    {   HASH_BATCH_NORM_KEY( SRC_TYPE, OUT_TYPE, 1), \
        HASH_BATCH_NORM_SH_KERNEL_2D_NAME( SRC_TYPE, OUT_TYPE), \
        VSI_NN_GEN_BATCH_NORM_KERNEL_SOURCE_NAME },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_BATCH_NORM_KERNELS(F32, F32)
    TENSOR_BATCH_NORM_KERNELS(F32, U8)
    TENSOR_BATCH_NORM_KERNELS(F32, I32)

    TENSOR_BATCH_NORM_KERNELS_2D(F32, F32)
    TENSOR_BATCH_NORM_KERNELS_2D(F32, U8)
    TENSOR_BATCH_NORM_KERNELS_2D(F32, I32)

    TENSOR_BATCH_NORM_KERNELS(U8,  U8)
    TENSOR_BATCH_NORM_KERNELS(U8,  F32)
    TENSOR_BATCH_NORM_KERNELS(I32, I32)
    TENSOR_BATCH_NORM_KERNELS(I32, F32)

    TENSOR_BATCH_NORM_KERNELS_2D(U8,  U8)
    TENSOR_BATCH_NORM_KERNELS_2D(U8,  F32)
    TENSOR_BATCH_NORM_KERNELS_2D(I32, I32)
    TENSOR_BATCH_NORM_KERNELS_2D(I32, F32)
};

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)
#define SCALAR_INPUT_EPS            (6)
#define SCALAR_INPUT_SCALE          (7)
#define SCALAR_INPUT_TAIL           (8)
#define SCALAR_OUTPUT_SCALE         (9)
#define SCALAR_OUTPUT_ZP            (10)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_log_softmax_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    gpu_param_t gpu_param = {
        3,         // workdim
        {0, 0, 0}, // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0}, // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0}, // localWorkSize: local group size in thread
        {0, 0, 0}  // globalWorkSize: image size in thread
        };

    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * in_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    in_shape  = attr[0]->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = in_shape->data[0];
    gpu_param.global_size[1] = in_shape->data[1];
    gpu_param.global_size[2] = in_shape->size > 2 ? in_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
    }

    return status;
} /* _log_softmax_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    if (input_dtype == I8 || input_dtype == I16)
    {
        input_dtype = I32;
    }
    else if (input_dtype == F16)
    {
        input_dtype = F32;
    }

    if (output_dtype == I8 || output_dtype == I16)
    {
        output_dtype = I32;
    }
    else if (output_dtype == F16)
    {
        output_dtype = F32;
    }

    key = HASH_BATCH_NORM_KEY( input_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _log_softmax_initializer;
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
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float input_tail = (float)vsi_nn_get_tensor_zero_point(inputs[0]) * input_scale;
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float output_zp = (float)vsi_nn_get_tensor_zero_point(outputs[0]) + 0.5f;
    float eps = vsi_nn_kernel_param_get_float32(params, "eps");

    if ( (inputs[1]->attr.is_const && inputs[2]->attr.is_const)
        || ( inputs[1]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16
          && inputs[1]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32 )
        || ( inputs[2]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16
          && inputs[2]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32)
        || ( inputs[3]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16
          && inputs[3]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32)
        || ( inputs[4]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16
          && inputs[4]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32 )
        )
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     )
    {
        return NULL;
    }

    image_2d = ((inputs[0]->attr.dim_num == 2) || (inputs[0]->attr.size[2] == 1));
    status = _query_kernel( inputs, outputs, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if( node )
        {
            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    inputs, 5, outputs, 1 );

            node_params[SCALAR_INPUT_EPS] = vsi_nn_kernel_scalar_create(
                    graph, F32, &eps );
            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input_scale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input_tail );
            node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &output_scale );
            node_params[SCALAR_OUTPUT_ZP] = vsi_nn_kernel_scalar_create(
                    graph, F32, &output_zp );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_EPS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( batchnorm_single, _setup )
