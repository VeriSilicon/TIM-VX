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

#define MOD_KERNEL_SOURCE_NAME "mod"

#define MOD_HASH_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))


#define MOD_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { MOD_HASH_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
      CVIVANTE_NAMESPACE("cl.mod_"#IN0_TYPE#IN1_TYPE"to"#OUT_TYPE), \
      MOD_KERNEL_SOURCE_NAME},

#define MOD_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { MOD_HASH_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
      CVIVANTE_NAMESPACE("cl.mod_"#IN0_TYPE#IN1_TYPE"to"#OUT_TYPE"_2D"), \
      MOD_KERNEL_SOURCE_NAME },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _mod_kernel_map[] =
{

// Register kernel here
    MOD_KERNELS( F32, F32, F32 )
    MOD_KERNELS( I32, I32, I32 )
    MOD_KERNELS( I32, I32, U8 )
    MOD_KERNELS( U8,  U8,  U8 )
    MOD_KERNELS( U8,  I32, U8 )

    MOD_KERNELS_2D( F32, F32, F32 )
    MOD_KERNELS_2D( I32, I32, I32 )
    MOD_KERNELS_2D( I32, I32, U8 )
    MOD_KERNELS_2D( U8,  U8,  U8 )
    MOD_KERNELS_2D( U8,  I32, U8 )
};

/*
 * Kernel params
 */
static vx_param_description_t _mod_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MOD_PARAM_NUM  _cnt_of_array( _mod_kernel_param_def )
#define MOD_QUANT_PARAM_NUM   _cnt_of_array( _mod_kernel_param_def )
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_mod_initializer)
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
    vsi_status status = VSI_FAILURE;
    vx_tensor  output              = (vx_tensor)param[2];
    vsi_nn_kernel_tensor_attr_t *output_attr  = NULL;
    vsi_size_array_t             *output_shape = NULL;

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output );
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_shape = output_attr->shape;

    gpu_param.dim = output_shape->size < 3 ? 2 : 3;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] +  gpu_param.global_scale[0] - 1)
                                        /  gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (output_shape->data[1] +  gpu_param.global_scale[1] - 1)
                                        /  gpu_param.global_scale[1];
    gpu_param.global_size[2]   = output_shape->size > 2 ? output_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }

    return status;
} /* _mod_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
     const _kernel_map_type * kernel_map = _mod_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _mod_kernel_map );
    vx_param_description_t * param_def  = _mod_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _mod_kernel_param_def );
    vx_kernel_initialize_f  initializer = _mod_initializer;

    uint32_t key = 0;
    uint32_t i = 0;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in0_dtype)
    {
        in0_dtype = F32;
    }
    else if (I16 == in0_dtype || I8 == in0_dtype)
    {
        in0_dtype = I32;
    }

    if (F16 == in1_dtype)
    {
        in1_dtype = F32;
    }
    else if (I16 == in1_dtype || I8 == in1_dtype)
    {
        in1_dtype = I32;
    }

    if (F16 == out_dtype)
    {
        out_dtype  = F32;
    }
    else if (I16 == out_dtype || I8 == out_dtype)
    {
        out_dtype = I32;
    }

    key = MOD_HASH_KEY( in0_dtype, in1_dtype, out_dtype, image_2d);

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < (uint32_t)kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
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
    vsi_nn_kernel_node_param_t node_params[_MOD_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    float outputScale  = vsi_nn_get_tensor_scale(outputs[0]);
    float outputTail   = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float input0Scale  = vsi_nn_get_tensor_scale(inputs[0]);
    float input0Tail   = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float input1Scale  = vsi_nn_get_tensor_scale(inputs[1]);
    float input1Tail   = (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    int32_t isfmod = vsi_nn_kernel_param_get_int32(params, "isfmod");

    outputScale = 1.0f / outputScale;
    input0Tail   = -(input0Tail * input0Scale);
    input1Tail   = -(input1Tail * input1Scale);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2);

    status = _query_kernel( kernel, inputs, outputs, image_2d);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            size_t node_params_num = MOD_QUANT_PARAM_NUM;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _MOD_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[3] = vsi_nn_kernel_scalar_create( graph, I32, &isfmod );
            node_params[4] = vsi_nn_kernel_scalar_create( graph, F32, &input0Scale );
            node_params[5] = vsi_nn_kernel_scalar_create(graph, F32, &input0Tail );
            node_params[6] = vsi_nn_kernel_scalar_create( graph, F32, &input1Scale );
            node_params[7] = vsi_nn_kernel_scalar_create(graph, F32, &input1Tail );
            node_params[8] = vsi_nn_kernel_scalar_create( graph, F32, &outputScale );
            node_params[9] = vsi_nn_kernel_scalar_create(graph, F32, &outputTail );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( mod, _setup )

