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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

#define KERNEL_SOURCE_1    "maxpoolwithargmax"
#define KERNEL_SOURCE_2    "maxpoolwithargmax_2d"

// Add kernel hashtable here
#define MAXPOOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT_DTYPE0, OUT_DTYPE1, _image_2d) \
        (( IN_DTYPE << 24 ) | ( OUT_DTYPE0 << 20) | ( OUT_DTYPE1 << 12) | (_image_2d))

#define HASH_MAXPOOLWITHARGMAX_KERNELS( IN_DTYPE, OUT_DTYPE0, OUT_DTYPE1) \
        { MAXPOOLWITHARGMAX_HASH_KEY(IN_DTYPE, OUT_DTYPE0, OUT_DTYPE1, 0), \
        CVIVANTE_NAMESPACE("cl.maxpoolwithargmax_"#IN_DTYPE"to"#OUT_DTYPE0"_"#OUT_DTYPE1), \
        KERNEL_SOURCE_1 },

#define HASH_MAXPOOLWITHARGMAX_KERNELS_2D( IN_DTYPE, OUT_DTYPE0, OUT_DTYPE1) \
        { MAXPOOLWITHARGMAX_HASH_KEY(IN_DTYPE, OUT_DTYPE0, OUT_DTYPE1, 1), \
        CVIVANTE_NAMESPACE("cl.maxpoolwithargmax_"#IN_DTYPE"to"#OUT_DTYPE0"_"#OUT_DTYPE1"_2D"), \
        KERNEL_SOURCE_2 },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } maxpoolwithargmax_map[] =
{
    HASH_MAXPOOLWITHARGMAX_KERNELS(F32,  F32,  I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS(BF16, BF16, I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS(U32,  U32,   I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS(I32,  I32,  I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS_2D(F32,  F32,  I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS_2D(BF16, BF16, I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS_2D(U32,  U32,   I32)
    HASH_MAXPOOLWITHARGMAX_KERNELS_2D(I32,  I32,  I32)
};

/*
 * Kernel params
 */
static vx_param_description_t _maxpoolwithargmax_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MAXPOOLWITHARGMAX_PARAM_NUM  _cnt_of_array( _maxpoolwithargmax_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_maxpoolwithargmax_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_status   status             = VSI_FAILURE;
    vx_tensor    output             = (vx_tensor)param[1];
    vsi_nn_kernel_tensor_attr_t * attr_out = NULL;
    vsi_size_array_t * out_shape   = NULL;

    VSI_UNREFERENCED(param_size);

    attr_out = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output );
    CHECK_PTR_FAIL_GOTO( attr_out, "vsi_nn_kernel_tensor_attr_create fail.", final );

    out_shape = attr_out->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((out_shape->data[0] +  gpu_param.global_scale[0] - 1)
                                        /  gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = out_shape->data[1];
    gpu_param.global_size[2]   = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (attr_out)
    {
        vsi_nn_kernel_tensor_attr_release(&attr_out);
    }

    return status;
} /* _maxpoolwithargmax_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t is_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input_dtype   = U8;
    vsi_nn_kernel_dtype_e output0_dtype = U8;
    vsi_nn_kernel_dtype_e output1_dtype = I32;
    uint32_t key = 0;
    size_t i = 0;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output0_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    output1_dtype = vsi_nn_kernel_map_dtype( outputs[1]->attr.dtype.vx_type );

    if (input_dtype == U8)
    {
        input_dtype = U32;
    }

    if (input_dtype == I8 || input_dtype == I16)
    {
        input_dtype = I32;
    }

    if (input_dtype == F16)
    {
        input_dtype = F32;
    }

    if (output0_dtype == U8)
    {
        output0_dtype = U32;
    }

    if (output0_dtype == I8 || output0_dtype == I16)
    {
        output0_dtype = I32;
    }

    if (output0_dtype == F16)
    {
        output0_dtype = F32;
    }

    key = MAXPOOLWITHARGMAX_HASH_KEY( input_dtype, output0_dtype, output1_dtype, is_2d);

    for ( i = 0; i < _cnt_of_array(maxpoolwithargmax_map); i ++ )
    {
        if ( maxpoolwithargmax_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(maxpoolwithargmax_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  maxpoolwithargmax_map[i].function_name );
        kernel->info.parameters = _maxpoolwithargmax_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _maxpoolwithargmax_kernel_param_def );
        kernel->info.initialize = _maxpoolwithargmax_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                maxpoolwithargmax_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                maxpoolwithargmax_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_MAXPOOLWITHARGMAX_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t ksize_x  = vsi_nn_kernel_param_get_int32(params, "ksize_x");
    int32_t ksize_y  = vsi_nn_kernel_param_get_int32(params, "ksize_y");
    int32_t stride_x = vsi_nn_kernel_param_get_int32(params, "stride_x");
    int32_t stride_y = vsi_nn_kernel_param_get_int32(params, "stride_y");
    int32_t pad_x    = vsi_nn_kernel_param_get_int32(params, "pad_left");
    int32_t pad_y    = vsi_nn_kernel_param_get_int32(params, "pad_top");
    int32_t image_2d = inputs[0]->attr.dim_num == 2 ? 1 : 0;
    int32_t width    = (int32_t)inputs[0]->attr.size[0];
    int32_t height   = (int32_t)inputs[0]->attr.size[1];
    float   outputScale  = vsi_nn_get_tensor_scale(outputs[0]);
    float   outputTail   = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float   inputScale   = vsi_nn_get_tensor_scale(inputs[0]);
    float   inputTail    = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float   scale_value  = 1.0f;
    float   tail_value   = 0.0f;

    if ( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[1]->attr.size,
                outputs[1]->attr.dim_num ))
    {
        return NULL;
    }

    scale_value = inputScale / outputScale;
    tail_value  = outputTail - inputTail * inputScale / outputScale;

    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 3;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _MAXPOOLWITHARGMAX_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &height );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_value );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &tail_value );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _MAXPOOLWITHARGMAX_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( maxpoolwithargmax, _setup )

