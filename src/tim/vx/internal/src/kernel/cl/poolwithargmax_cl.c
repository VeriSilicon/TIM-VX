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


#define _POOLWITHARGMAX_KERNEL_SOURCE      "poolwithargmax"

#define POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, _image_2d ) \
        ((IN_DTYPE << 20) | (OUT0_DTYPE << 12) | (OUT1_DTYPE << 4) | (_image_2d))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE ) \
        { POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, 0 ), \
          CVIVANTE_NAMESPACE("cl.poolwithargmax_"#IN_DTYPE"to_"#OUT0_DTYPE"_"#OUT1_DTYPE), \
          _POOLWITHARGMAX_KERNEL_SOURCE }

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE ) \
        { POOLWITHARGMAX_HASH_KEY( IN_DTYPE, OUT0_DTYPE, OUT1_DTYPE, 1 ), \
          CVIVANTE_NAMESPACE("cl.poolwithargmax_"#IN_DTYPE"to_"#OUT0_DTYPE"_"#OUT1_DTYPE"_2D"), \
          _POOLWITHARGMAX_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _poolwithargmax_kernel_map[] =
{
    PACK_KERNEL_MAP( F32, F32, U8 ),
    PACK_KERNEL_MAP( F32, U8,  U8 ),
    PACK_KERNEL_MAP( U8,  F32, U8 ),
    PACK_KERNEL_MAP( U8,  U8,  U8 ),
    PACK_KERNEL_MAP( I32, I32, U8 ),
    PACK_KERNEL_MAP_2D( F32, F32, U8 ),
    PACK_KERNEL_MAP_2D( F32, U8,  U8 ),
    PACK_KERNEL_MAP_2D( U8,  F32, U8 ),
    PACK_KERNEL_MAP_2D( U8,  U8,  U8 ),
    PACK_KERNEL_MAP_2D( I32, I32, U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _poolwithargmax_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _POOLWITHARGMAX_PARAM_NUM  _cnt_of_array( _poolwithargmax_kernel_param_def )

#define SCALAR_SCALE          (3)
#define SCALAR_TAIL           (4)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_poolwithargmax_initializer)
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
    vsi_bool          image_2d    = FALSE;

    VSI_UNREFERENCED(param_size);

    attr_out = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output );
    CHECK_PTR_FAIL_GOTO( attr_out, "vsi_nn_kernel_tensor_attr_create fail.", final );

    out_shape = attr_out->shape;
    image_2d = (vsi_bool)(out_shape->size < 3 || 1 == out_shape->data[2]);

    gpu_param.dim = image_2d ? 2 : 3;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((out_shape->data[0] +  gpu_param.global_scale[0] - 1)
                                        /  gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (out_shape->data[1] +  gpu_param.global_scale[1] - 1)
                                        /  gpu_param.global_scale[1];
    gpu_param.global_size[2]   = image_2d ? 1 : out_shape->data[2];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (attr_out)
    {
        vsi_nn_kernel_tensor_attr_release(&attr_out);
    }

    return status;
} /* _poolwithargmax_initializer() */



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
    vsi_nn_kernel_dtype_e out0_dtype;
    vsi_nn_kernel_dtype_e out1_dtype;
    const _kernel_map_type * kernel_map = _poolwithargmax_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _poolwithargmax_kernel_map );
    vx_param_description_t * param_def  = _poolwithargmax_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _poolwithargmax_kernel_param_def );
    vx_kernel_initialize_f  initializer = _poolwithargmax_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype   = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out0_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    out1_dtype = vsi_nn_kernel_map_dtype( outputs[1]->attr.dtype.vx_type );

    if (F16 == in_dtype)
    {
        in_dtype   = F32;
    }

    if (F16 == out0_dtype)
    {
        out0_dtype   = F32;
    }

    if ((U8 != in_dtype) && (U8 != out0_dtype))
    {
        *is_use_u8_kernel = FALSE;
        param_def_size    = param_def_size - 2;
    }
    else
    {
        *is_use_u8_kernel = TRUE;
    }

    key = POOLWITHARGMAX_HASH_KEY( in_dtype, out0_dtype, out1_dtype, image_2d );

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
    vsi_nn_kernel_node_param_t node_params[_POOLWITHARGMAX_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t  ksize_x  = 0;
    int32_t  ksize_y  = 0;
    int32_t  stride_x = 0;
    int32_t  stride_y = 0;
    int32_t  pad_x    = 0;
    int32_t  pad_y    = 0;
    vsi_bool image_2d = FALSE;
    float    outputScale  = vsi_nn_get_tensor_scale(outputs[0]);
    float    outputTail   = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float    inputScale   = vsi_nn_get_tensor_scale(inputs[0]);
    float    inputTail    = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float    scale_value  = 1.0f;
    float    tail_value   = 0.0f;
    vsi_bool is_use_u8_kernel = FALSE;

    ksize_x  = vsi_nn_kernel_param_get_int32(params, "ksize_x");
    ksize_y  = vsi_nn_kernel_param_get_int32(params, "ksize_y");
    stride_x = vsi_nn_kernel_param_get_int32(params, "stride_x");
    stride_y = vsi_nn_kernel_param_get_int32(params, "stride_y");
    pad_x    = vsi_nn_kernel_param_get_int32(params, "pad_x");
    pad_y    = vsi_nn_kernel_param_get_int32(params, "pad_y");

    if ((2 != ksize_x) || (2 != ksize_y) || (2 != stride_x) || (2 != stride_y) || (0 != pad_x) || (0 != pad_y))
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
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

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d, &is_use_u8_kernel);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            size_t node_params_num = _POOLWITHARGMAX_PARAM_NUM - 2;

            if (is_use_u8_kernel)
            {
                node_params[SCALAR_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &scale_value );
                node_params[SCALAR_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &tail_value );
                node_params_num = _POOLWITHARGMAX_PARAM_NUM;
            }

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, node_params_num,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            if (is_use_u8_kernel)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_TAIL] );
            }
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( poolwithargmax, _setup )
