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
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

#define _RELU_KERAS_KERNEL_SOURCE      "relu_keras"

#define STR(a) #a
// Add kernel hashtable here
#define RELU_KERAS_HASH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (_image_2d))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { RELU_KERAS_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0 ), \
          CVIVANTE_NAMESPACE("cl.relu_keras_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
          _RELU_KERAS_KERNEL_SOURCE }

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE ) \
        { RELU_KERAS_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1 ), \
          CVIVANTE_NAMESPACE("cl.relu_keras_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
          _RELU_KERAS_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _relu_keras_kernel_map[] =
{
    PACK_KERNEL_MAP(F32, F32),
    PACK_KERNEL_MAP(F32, U8),
    PACK_KERNEL_MAP(U8,  U8),
    PACK_KERNEL_MAP(U8,  F32),
    PACK_KERNEL_MAP_2D(F32, F32),
    PACK_KERNEL_MAP_2D(F32, U8),
    PACK_KERNEL_MAP_2D(U8,  U8),
    PACK_KERNEL_MAP_2D(U8,  F32),
};


/*
 * Kernel params
 */
static vx_param_description_t _relu_keras_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_ALPHA              (2)
#define SCALAR_MAX_VALUE          (3)
#define SCALAR_THRESHOLD          (4)
#define SCALAR_OFFSET             (5)
#define SCALAR_INPUT_SCALE        (6)
#define SCALAR_INPUT_TAIL         (7)
#define SCALAR_OUTPUT_SCALE       (8)
#define SCALAR_OUTPUT_TAIL        (9)

#define _RELU_KERAS_PARAM_NUM         6
#define _RELU_KERAS_QUANT_PARAM_NUM   _cnt_of_array( _relu_keras_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_relu_keras_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_size_array_t * out_shape                 = NULL;

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    out_shape  = output_attr->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.dim = (out_shape->size < 3 || 1 == out_shape->data[2]) ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    return status;
} /* _relu_keras_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d,
    vsi_bool *is_use_u8_kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _relu_keras_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _relu_keras_kernel_map );
    vx_param_description_t * param_def  = _relu_keras_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _relu_keras_kernel_param_def );
    vx_kernel_initialize_f  initializer = _relu_keras_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in_dtype)
    {
        in_dtype = F32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }

   if ((U8 == in_dtype) || (U8 == out_dtype))
    {
        param_def_size    = _RELU_KERAS_QUANT_PARAM_NUM;
        *is_use_u8_kernel = TRUE;
    }
    else
    {
        param_def_size    = _RELU_KERAS_PARAM_NUM;
        *is_use_u8_kernel = FALSE;
    }

    key = RELU_KERAS_HASH_KEY( in_dtype, (uint32_t)out_dtype, image_2d );

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                kernel_map[i].source_name );
        // Register binary source
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
    vsi_nn_kernel_node_param_t node_params[_RELU_KERAS_QUANT_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    float    outputScale  = 1.0f;
    float    outputTail   = 0.0f;
    float    inputScale   = 1.0f;
    float    inputTail    = 0.0f;
    vsi_bool is_use_u8_kernel = FALSE;
    float    alpha        = vsi_nn_kernel_param_get_float32( params, "alpha" );
    float    max_value    = vsi_nn_kernel_param_get_float32( params, "max_value" );
    float    threshold    = vsi_nn_kernel_param_get_float32( params, "threshold" );
    float    offset       = -alpha * threshold;

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if (VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC == inputs[0]->attr.dtype.qnt_type)
    {
        inputScale   = inputs[0]->attr.dtype.scale == 0.0f ? 1.0f : inputs[0]->attr.dtype.scale;
        inputTail    = -((float)inputs[0]->attr.dtype.zero_point * inputScale);
    }

    if (VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC == outputs[0]->attr.dtype.qnt_type)
    {
        outputScale  = outputs[0]->attr.dtype.scale == 0.0f ? 1.0f : outputs[0]->attr.dtype.scale;
        outputScale = 1.0f / outputScale;
        outputTail   = (float)outputs[0]->attr.dtype.zero_point;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d, &is_use_u8_kernel );

    if( VSI_SUCCESS == status)
    {
        size_t node_params_num = _RELU_KERAS_PARAM_NUM;

        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _RELU_KERAS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_ALPHA]     = vsi_nn_kernel_scalar_create( graph, F32, &alpha );
            node_params[SCALAR_MAX_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &max_value );
            node_params[SCALAR_THRESHOLD] = vsi_nn_kernel_scalar_create( graph, F32, &threshold );
            node_params[SCALAR_OFFSET]    = vsi_nn_kernel_scalar_create( graph, F32, &offset );
           if (is_use_u8_kernel)
            {
                node_params[SCALAR_INPUT_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &inputScale );
                node_params[SCALAR_INPUT_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &inputTail );
                node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &outputScale );
                node_params[SCALAR_OUTPUT_TAIL]  = vsi_nn_kernel_scalar_create(graph, F32, &outputTail );
                node_params_num = _RELU_KERAS_QUANT_PARAM_NUM;
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_ALPHA] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_MAX_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_THRESHOLD] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OFFSET] );
            if (is_use_u8_kernel)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_TAIL] );
            }
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( relu_keras, _setup )

