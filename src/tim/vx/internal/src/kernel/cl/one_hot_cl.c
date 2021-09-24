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
#include "utils/vsi_nn_dtype_util.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_ONE_HOT,
} _internal_kernel_e;

#define _ONE_HOT_KERNEL_SOURCE      "one_hot"

// Add kernel hashtable here
#define HASH_ONE_HOT_SH_KERNEL_NAME(SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.one_hot_"#SRC_TYPE"to"#DST_TYPE)

#define ONE_HOT_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))

#define PACK_ONE_HOT_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
{ ONE_HOT_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
  HASH_ONE_HOT_SH_KERNEL_NAME( IN_DTYPE, OUT_DTYPE ), \
  _ONE_HOT_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _one_hot_kernel_map[] =
{
    // Register kernel here
    PACK_ONE_HOT_KERNEL_MAP( F32, F32 ),
    PACK_ONE_HOT_KERNEL_MAP( I32, I32 ),
    PACK_ONE_HOT_KERNEL_MAP( I32, F32 ),
    PACK_ONE_HOT_KERNEL_MAP( I32, U8 ),
    PACK_ONE_HOT_KERNEL_MAP( U8,  U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _one_hot_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define SCALAR_INPUT_DEPTH           (2)
#define SCALAR_INPUT_ON_VALUE        (3)
#define SCALAR_INPUT_OFF_VALUE       (4)
#define SCALAR_INPUT_SCALE           (5)
#define SCALAR_INPUT_TAIL            (6)
#define _ONE_HOT_PARAM_NUM  _cnt_of_array( _one_hot_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_one_hot_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    gpu_param_t gpu_param = {
        2,         // workdim
        {0, 0, 0}, // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0}, // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0}, // localWorkSize: local group size in thread
        {0, 0, 0}  // globalWorkSize: image size in thread
        };

    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * in_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    in_shape  = attr[0]->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (in_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = in_shape->data[1];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if ( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _one_hot_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _one_hot_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _one_hot_kernel_map );
    vx_param_description_t * param_def  = _one_hot_kernel_param_def;
    vx_kernel_initialize_f  initializer = _one_hot_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (in_dtype == F16)
    {
        in_dtype = F32;
    }

    if (out_dtype == F16)
    {
        out_dtype = F32;
    }
    else if (out_dtype == I16 || out_dtype == I8)
    {
        out_dtype = I32;
    }

    key = ONE_HOT_HASH_KEY( in_dtype, out_dtype );

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
        kernel->info.numParams   = _cnt_of_array( _one_hot_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_ONE_HOT_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* rs_tensors[2] = { NULL };
    vsi_size_t shape[2][VSI_NN_MAX_DIM_NUM] = {{ 0 }};
    int32_t i = 0;
    vsi_size_t num_elements = vsi_nn_vxGetTensorElementNum(&inputs[0]->attr);
    vsi_size_t prefix_dim_size = 1;
    vsi_size_t suffix_dim_size = 0;
    int32_t depth = vsi_nn_kernel_param_get_int32( params, "depth" );
    vsi_nn_kernel_dtype_e out_dtype;
    uint32_t data[2] = {0};
    float on_value = vsi_nn_kernel_param_get_float32( params, "on_value" );
    float off_value = vsi_nn_kernel_param_get_float32( params, "off_value" );
    int32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );
    float inputScale = inputs[0]->attr.dtype.scale;
    float inputTail = (float)inputs[0]->attr.dtype.zero_point * inputScale;

    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (out_dtype != F32 && out_dtype != F16)
    {
        vsi_nn_Float32ToDtype(on_value, (uint8_t*)&data[0], &outputs[0]->attr.dtype);
        vsi_nn_Float32ToDtype(off_value, (uint8_t*)&data[1], &outputs[0]->attr.dtype);
    }
    else
    {
        data[0] = *(uint32_t*)&on_value;
        data[1] = *(uint32_t*)&off_value;
    }

    axis = axis == -1 ? (int32_t)inputs[0]->attr.dim_num : (int32_t)inputs[0]->attr.dim_num - axis;
    for (i = 0; i < axis; i++)
    {
        prefix_dim_size *= inputs[0]->attr.size[i];
    }

    suffix_dim_size = num_elements / prefix_dim_size;

    shape[0][0] = suffix_dim_size;
    shape[0][1] = prefix_dim_size;
    shape[1][0] = suffix_dim_size;
    shape[1][1] = depth;
    shape[1][2] = prefix_dim_size;

    rs_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shape[0], 2 );
    rs_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shape[1], 3 );

    if ( !vsi_nn_kernel_gpu_check_shape( rs_tensors[1]->attr.size,
                rs_tensors[1]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ONE_HOT_PARAM_NUM,
                    &rs_tensors[0], input_num, &rs_tensors[1], output_num );
            node_params[SCALAR_INPUT_DEPTH] = vsi_nn_kernel_scalar_create(
                graph, I32, &depth );
            node_params[SCALAR_INPUT_ON_VALUE] = vsi_nn_kernel_scalar_create(
                graph, U32, &data[0] );
            node_params[SCALAR_INPUT_OFF_VALUE] = vsi_nn_kernel_scalar_create(
                graph, U32, &data[1] );
            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputScale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputTail );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ONE_HOT_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, final );
        }
    }

final:
    if (rs_tensors[0])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[0] );
    }

    if (rs_tensors[1])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[1] );
    }

    for (i = SCALAR_INPUT_DEPTH; i < _ONE_HOT_PARAM_NUM; i++)
    {
        if (node_params[i])
        {
            vsi_nn_kernel_scalar_release( &node_params[i] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( one_hot, _setup )
