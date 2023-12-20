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

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _CPU_IO_NUM         (_INPUT_NUM + _OUTPUT_NUM)
#define _ADD_MEAN_STD_NORM_KERNEL_SOURCE      "add_mean_std_norm"


// Add kernel hashtable here
#define ADD_MEAN_STD_NORM_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        ((IN0_DTYPE << 20) | ( IN1_DTYPE << 12 ) | ( OUT_DTYPE << 4) )

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { ADD_MEAN_STD_NORM_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), \
        CVIVANTE_NAMESPACE("cl.add_mean_std_norm_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE), \
         _ADD_MEAN_STD_NORM_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _add_mean_std_norm_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, F32 ),
    PACK_KERNEL_MAP( U8 , U8 , F32 ),
    PACK_KERNEL_MAP( U8 , U8 , U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _add_mean_std_norm_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _ADD_MEAN_STD_NORM_PARAM_NUM  _cnt_of_array( _add_mean_std_norm_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_add_mean_std_norm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vx_tensor                    input0      = (vx_tensor)param[0];
    vsi_nn_kernel_tensor_attr_t *input0_attr = NULL;
    vsi_size_array_t             *input_shape = NULL;

    VSI_UNREFERENCED(param_size);

    input0_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input0);
    CHECK_PTR_FAIL_GOTO( input0_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    input_shape   = input0_attr->shape;

    gpu_param.global_offset[0] = 0;
    gpu_param.global_offset[1] = 0;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.local_size[0]    = 16;
    gpu_param.local_size[1]    = 1;
    gpu_param.global_size[0]   = 16;
    gpu_param.global_size[1]   = gpu_align_p2( (input_shape->data[1] + gpu_param.global_scale[1] - 1)
                                             / gpu_param.global_scale[1], gpu_param.local_size[1] );

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (input0_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input0_attr);
    }
    return status;
} /* _add_mean_std_norm_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _add_mean_std_norm_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _add_mean_std_norm_kernel_map );
    vx_param_description_t * param_def  = _add_mean_std_norm_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _add_mean_std_norm_kernel_param_def );
    vx_kernel_initialize_f  initializer = _add_mean_std_norm_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in0_dtype)
    {
        in0_dtype = F32;
    }

    if (F16 == in1_dtype)
    {
        in1_dtype = F32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }

    key = ADD_MEAN_STD_NORM_HASH_KEY( in0_dtype, in1_dtype, out_dtype );

    for( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < (uint32_t)kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_ADD_MEAN_STD_NORM_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float   eps         = vsi_nn_kernel_param_get_float32( params, "eps" );
    float   rsEps       = (float)(1.0f / sqrtf(eps));
    float   dimRatio    = (float)(1.0f / (inputs[0]->attr.size[0]));
    float   input0Scale = vsi_nn_get_tensor_scale(inputs[0]);
    float   input0Tail  = (float)vsi_nn_get_tensor_zero_point(inputs[0]) * input0Scale;
    float   input1Scale = vsi_nn_get_tensor_scale(inputs[1]);
    float   input1Tail  = (float)vsi_nn_get_tensor_zero_point(inputs[1]) * input1Scale;
    float   outputScale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float   outputZP    = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    int32_t width       = (int32_t)inputs[0]->attr.size[0];

    status = _query_kernel( kernel, inputs, outputs );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U32 = 0;
            border.constant_value.S16 = 0;
            border.constant_value.U8 = 0;
            if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
            {
                border.constant_value.U8 = (uint8_t)vsi_nn_get_tensor_zero_point(inputs[0]);
            }
            status  = vsi_nn_kernel_node_set_border( node, &border );
            VSI_ASSERT( status == VSI_SUCCESS );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ADD_MEAN_STD_NORM_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[_CPU_IO_NUM]     = vsi_nn_kernel_scalar_create( graph, F32, &rsEps );
            node_params[_CPU_IO_NUM + 1] = vsi_nn_kernel_scalar_create( graph, F32, &dimRatio );
            node_params[_CPU_IO_NUM + 2] = vsi_nn_kernel_scalar_create( graph, F32, &input0Scale );
            node_params[_CPU_IO_NUM + 3] = vsi_nn_kernel_scalar_create( graph, F32, &input0Tail );
            node_params[_CPU_IO_NUM + 4] = vsi_nn_kernel_scalar_create( graph, F32, &input1Scale );
            node_params[_CPU_IO_NUM + 5] = vsi_nn_kernel_scalar_create( graph, F32, &input1Tail );
            node_params[_CPU_IO_NUM + 6] = vsi_nn_kernel_scalar_create( graph, F32, &outputScale );
            node_params[_CPU_IO_NUM + 7] = vsi_nn_kernel_scalar_create( graph, F32, &outputZP );
            node_params[_CPU_IO_NUM + 8] = vsi_nn_kernel_scalar_create( graph, I32, &width );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ADD_MEAN_STD_NORM_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 1] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 2] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 3] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 4] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 5] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 6] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 7] );
            vsi_nn_kernel_scalar_release( &node_params[_CPU_IO_NUM + 8] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( add_mean_std_norm, _setup )
