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

#define HASH_REDUCEPROD_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, _image_2d) \
    ((AXIS << 20) | (IN_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

 #define HASH_REDUCEPROD_KERNEL_SOURCE_NAME(AXIS) \
    "reduceprod_internal_axis"#AXIS

#define HASH_REDUCEPROD_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_REDUCEPROD_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0), \
        CVIVANTE_NAMESPACE("cl.reduceprod_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE), \
        HASH_REDUCEPROD_KERNEL_SOURCE_NAME(AXIS) },

#define HASH_REDUCEPROD_KERNELS_2D( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_REDUCEPROD_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("cl.reduceprod_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE"_2D"), \
        HASH_REDUCEPROD_KERNEL_SOURCE_NAME(AXIS) },



typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _reduceprod_internal_kernel_map[] =
{
    HASH_REDUCEPROD_KERNELS( 0, F32, F32 )
    HASH_REDUCEPROD_KERNELS( 0, I32, I32 )
    HASH_REDUCEPROD_KERNELS( 0, U8,  U8 )
    HASH_REDUCEPROD_KERNELS( 1, F32, F32 )
    HASH_REDUCEPROD_KERNELS( 1, I32, I32 )
    HASH_REDUCEPROD_KERNELS( 1, U8,  U8 )
    HASH_REDUCEPROD_KERNELS( 2, F32, F32 )
    HASH_REDUCEPROD_KERNELS( 2, I32, I32 )
    HASH_REDUCEPROD_KERNELS( 2, U8,  U8 )

    HASH_REDUCEPROD_KERNELS_2D( 0, F32, F32 )
    HASH_REDUCEPROD_KERNELS_2D( 0, I32, I32 )
    HASH_REDUCEPROD_KERNELS_2D( 0, U8,  U8 )
    HASH_REDUCEPROD_KERNELS_2D( 1, F32, F32 )
    HASH_REDUCEPROD_KERNELS_2D( 1, I32, I32 )
    HASH_REDUCEPROD_KERNELS_2D( 1, U8,  U8 )
};


/*
 * Kernel params
 */
static vx_param_description_t _reduceprod_internal_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _REDUCEPROD_INTERNAL_PARAM_NUM  _cnt_of_array( _reduceprod_internal_kernel_param_def )

#define SCALAR_INPUT_SCALE          (2)
#define SCALAR_INPUT_TAIL           (3)
#define SCALAR_OUTPUT_SCALE         (4)
#define SCALAR_OUTPUT_TAIL          (5)

#define REDUCEPROD_PARAM_NUM         2
#define REDUCEPROD_QUANT_PARAM_NUM   _cnt_of_array( _reduceprod_internal_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_reduceprod_internal_initializer)
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
    vsi_nn_kernel_tensor_attr_t *output_attr   = NULL;
    vsi_size_array_t * output_shape             = NULL;

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    output_shape  = output_attr->shape;

    gpu_param.dim = 2;
    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (output_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (output_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );
    return status;
} /* _reduceprod_internal_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis,
    vsi_bool image_2d,
    vsi_bool *is_use_u8_kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _reduceprod_internal_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _reduceprod_internal_kernel_map );
    vx_param_description_t * param_def  = _reduceprod_internal_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _reduceprod_internal_kernel_param_def );
    vx_kernel_initialize_f  initializer = _reduceprod_internal_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in_dtype)
    {
        in_dtype  = F32;
    }

    if (F16 == out_dtype)
    {
        out_dtype  = F32;
    }

    if ((U8 == in_dtype) || (U8 == out_dtype))
    {
        param_def_size = REDUCEPROD_QUANT_PARAM_NUM;
        *is_use_u8_kernel = TRUE;
    }
    else
    {
        param_def_size = REDUCEPROD_PARAM_NUM;
        *is_use_u8_kernel = FALSE;
    }

    key = HASH_REDUCEPROD_HASH_KEY( axis, in_dtype, out_dtype, image_2d );

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
    vsi_nn_kernel_node_param_t node_params[_REDUCEPROD_INTERNAL_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    int32_t  axis = 0;
    float    outputScale  = outputs[0]->attr.dtype.scale == 0.0f ? 1.0f : outputs[0]->attr.dtype.scale;
    float    outputTail   = (float)outputs[0]->attr.dtype.zero_point;
    float    inputScale   = inputs[0]->attr.dtype.scale == 0.0f ? 1.0f : inputs[0]->attr.dtype.scale;
    float    inputTail    = (float)inputs[0]->attr.dtype.zero_point;
    vsi_bool is_use_u8_kernel = FALSE;

    outputScale = 1.0f / outputScale;
    inputTail   = -(inputTail * inputScale);

    axis = vsi_nn_kernel_param_get_int32(params, "axis");

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num )
     || axis > 2)
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, axis, image_2d, &is_use_u8_kernel);
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            size_t node_params_num = REDUCEPROD_PARAM_NUM;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _REDUCEPROD_INTERNAL_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            if (is_use_u8_kernel)
            {
                node_params[SCALAR_INPUT_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &inputScale );
                node_params[SCALAR_INPUT_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &inputTail );
                node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &outputScale );
                node_params[SCALAR_OUTPUT_TAIL]  = vsi_nn_kernel_scalar_create(graph, F32, &outputTail );
                node_params_num = REDUCEPROD_QUANT_PARAM_NUM;
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
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

REGISTER_BACKEND_CL( reduceprod_internal, _setup )

