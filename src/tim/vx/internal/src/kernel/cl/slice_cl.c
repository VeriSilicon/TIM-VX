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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

    /*
    * Define kernel meta.
    */
    typedef enum
{
    INTERNAL_KERNEL_SLICE,
} _internal_kernel_e;

#define _SLICE_KERNEL_SOURCE      "slice"
#define SLICE_SH_KERNEL_NAME(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    CVIVANTE_NAMESPACE("cl.slice_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE)

// Add kernel hashtable here
#define SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE , _IMAGE_2D) \
    (( IN1_DTYPE << 18 ) | ( IN0_DTYPE << 10 ) | ( OUT_DTYPE << 2 ) | (_IMAGE_2D))

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
{   SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0 ), \
    SLICE_SH_KERNEL_NAME( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), SOURCE }

#define SLICE_SH_KERNEL_2D_NAME(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    CVIVANTE_NAMESPACE("cl.slice_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE"_2D")

#define PACK_KERNEL_MAP_2D( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
{   SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1 ), \
    SLICE_SH_KERNEL_2D_NAME( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _slice_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, I32, F32, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I32, I32, I32, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U8,  I32, U8,  _SLICE_KERNEL_SOURCE ),

    PACK_KERNEL_MAP_2D( F32, I32, F32, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I32, I32, I32, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( U8,  I32, U8,  _SLICE_KERNEL_SOURCE ),
};

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)

/*
* Kernel params
*/
static vx_param_description_t _slice_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SLICE_PARAM_NUM  _cnt_of_array( _slice_kernel_param_def )
#define SCALAR_INPUT_SCALE          (3)
#define SCALAR_INPUT_TAIL           (4)
#define SCALAR_OUTPUT_SCALE         (5)
#define SCALAR_OUTPUT_ZP            (6)
/*
* Kernel initializer
*/
DEF_KERNEL_INITIALIZER(_slice_initializer)
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

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
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
} /* _slice_initializer() */

/*
* Query kernel
*/
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _slice_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _slice_kernel_map );
    vx_param_description_t * param_def  = _slice_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _slice_kernel_param_def );
    vx_kernel_initialize_f  initializer = _slice_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in0_dtype)
    {
        in0_dtype = F32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }

    key = SLICE_HASH_KEY( in0_dtype, in1_dtype, out_dtype, image_2d );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_SLICE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    uint32_t rank[_IO_NUM] = {0};
    vsi_size_t  shapes[_IO_NUM][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_tensor_t* reshape_tensors[_IO_NUM] = { NULL };
    int32_t i = 0;
    vsi_size_t input_batch = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    vsi_size_t output_batch = outputs[0]->attr.dim_num > 3 ? outputs[0]->attr.size[3] : 1;
    float inputScale = inputs[0]->attr.dtype.scale;
    float inputTail = (float)inputs[0]->attr.dtype.zero_point * inputScale;
    float outputScale = outputs[0]->attr.dtype.scale;
    float outputZP = (float)outputs[0]->attr.dtype.zero_point + 0.5f;

    outputScale = vsi_abs(outputScale) < 1e-5 ? 0.0f : 1.0f / outputScale;

    vsi_nn_kernel_optimize_1d_tensor_shape( (const vsi_size_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num,
        shapes[0], &rank[0]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const vsi_size_t*)inputs[1]->attr.size, inputs[1]->attr.dim_num,
        shapes[1], &rank[1]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const vsi_size_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num,
        shapes[2], &rank[2]);

    for (i = 0; i < _INPUT_NUM; i++)
    {
        reshape_tensors[i] = vsi_nn_reshape_tensor( graph,
            inputs[i], shapes[i], rank[i] );
    }
    reshape_tensors[_INPUT_NUM] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[_INPUT_NUM], rank[_INPUT_NUM] );

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[0]->attr.size,
        inputs[0]->attr.dim_num ) || input_batch != output_batch )
    {
        goto final;
    }

    image_2d = (rank[0] < 3 || shapes[0][2] == 1);

    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SLICE_PARAM_NUM,
                reshape_tensors, input_num, &reshape_tensors[_INPUT_NUM], output_num );
            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputScale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputTail );
            node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputScale );
            node_params[SCALAR_OUTPUT_ZP] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputZP );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SLICE_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
        }
    }

final:
    for (i = 0; i < _IO_NUM; i++)
    {
        vsi_safe_release_tensor(reshape_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

    REGISTER_BACKEND_CL( slice, _setup )
