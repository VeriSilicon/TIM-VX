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

#define _TOPK_KERNEL_SOURCE      "topk"
#define STR(a) #a
// Add kernel hashtable here
#define TOPK_HASH_KEY( IN_DTYPE, OUT_DTYPE, STAGES ) \
        ( ( IN_DTYPE ) | ( OUT_DTYPE << 8 ) | (STAGES << 16) )
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, STAGES ) \
        { TOPK_HASH_KEY( IN_DTYPE, OUT_DTYPE, STAGES ), \
          CVIVANTE_NAMESPACE("cl.topk_stage"STR(STAGES)"_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_I32"), \
          _TOPK_KERNEL_SOURCE }

#define TOPK_ODD_EVEN_SORT_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        ( ( IN_DTYPE ) | ( OUT_DTYPE << 8 ) )
#define PACK_ODD_EVEN_SORT_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { TOPK_ODD_EVEN_SORT_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
          CVIVANTE_NAMESPACE("cl.topk_odd_even_sort_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_I32"), \
          "topk_odd_even_sort" }

#define TOPK_ODD_EVEN_SORT_HASH_KEY2( IN_DTYPE, OUT_DTYPE ) \
       ( ( IN_DTYPE ) | ( OUT_DTYPE << 8 ) )
#define PACK_ODD_EVEN_SORT_KERNEL_MAP2( IN_DTYPE, OUT_DTYPE ) \
       { TOPK_ODD_EVEN_SORT_HASH_KEY2( IN_DTYPE, OUT_DTYPE ), \
         CVIVANTE_NAMESPACE("cl.topk_odd_even_sort_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_I32"), \
         "topk_odd_even_sort2" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _topk_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, 0 ),
    PACK_KERNEL_MAP( F32, F32, 1 ),
    PACK_KERNEL_MAP( F32, F32, 2 ),
    PACK_KERNEL_MAP( F32, F32, 3 ),
    PACK_KERNEL_MAP( F32, F32, 4 ),
    PACK_KERNEL_MAP( F32, F32, 5 ),
    PACK_KERNEL_MAP( F32, F32, 6 ),

    PACK_KERNEL_MAP( U32, U32, 0 ),
    PACK_KERNEL_MAP( U32, U32, 1 ),
    PACK_KERNEL_MAP( U32, U32, 2 ),
    PACK_KERNEL_MAP( U32, U32, 3 ),
    PACK_KERNEL_MAP( U32, U32, 4 ),
    PACK_KERNEL_MAP( U32, U32, 5 ),
    PACK_KERNEL_MAP( U32, U32, 6 ),

    PACK_KERNEL_MAP( I32, I32, 0 ),
    PACK_KERNEL_MAP( I32, I32, 1 ),
    PACK_KERNEL_MAP( I32, I32, 2 ),
    PACK_KERNEL_MAP( I32, I32, 3 ),
    PACK_KERNEL_MAP( I32, I32, 4 ),
    PACK_KERNEL_MAP( I32, I32, 5 ),
    PACK_KERNEL_MAP( I32, I32, 6 ),

    PACK_KERNEL_MAP( F32, U32, 0 ),
    PACK_KERNEL_MAP( F32, U32, 1 ),
    PACK_KERNEL_MAP( F32, U32, 2 ),
    PACK_KERNEL_MAP( F32, U32, 3 ),
    PACK_KERNEL_MAP( F32, U32, 4 ),
    PACK_KERNEL_MAP( F32, U32, 5 ),
    PACK_KERNEL_MAP( F32, U32, 6 ),

    PACK_KERNEL_MAP( F32, I32, 0 ),
    PACK_KERNEL_MAP( F32, I32, 1 ),
    PACK_KERNEL_MAP( F32, I32, 2 ),
    PACK_KERNEL_MAP( F32, I32, 3 ),
    PACK_KERNEL_MAP( F32, I32, 4 ),
    PACK_KERNEL_MAP( F32, I32, 5 ),
    PACK_KERNEL_MAP( F32, I32, 6 ),
};

static const _kernel_map_type _topk_odd_even_sort_kernel_map[] =
{
    // Register kernel here
    PACK_ODD_EVEN_SORT_KERNEL_MAP( F32, F32 ),
    PACK_ODD_EVEN_SORT_KERNEL_MAP( U32, U32 ),
    PACK_ODD_EVEN_SORT_KERNEL_MAP( I32, I32 ),
    PACK_ODD_EVEN_SORT_KERNEL_MAP2( F32, U32 ),
    PACK_ODD_EVEN_SORT_KERNEL_MAP2( F32, I32 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _topk_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _TOPK_PARAM_NUM  _cnt_of_array( _topk_kernel_param_def )
#define SCALAR_INPUT_NUM_STAGES (7)
#define SCALAR_INPUT_WIDTH      (8)

static vx_param_description_t _topk_odd_even_sort_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _TOPK_ODD_EVEN_SORT_PARAM_NUM  _cnt_of_array( _topk_odd_even_sort_kernel_param_def )
#define SCALAR_INPUT_SIZE  (9)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_topk_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * input_attr   = NULL;
    vsi_size_array_t * in_shape                = NULL;
    int32_t num_stages = 0;

    VSI_UNREFERENCED(param_size);

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_NUM_STAGES], &num_stages);
    CHECK_STATUS_FAIL_GOTO(status, final );

    in_shape  = input_attr->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.local_size[0]   = (size_t)(1 << num_stages);
    gpu_param.local_size[1]   = 1;
    gpu_param.global_size[0]  = (size_t)(1 << num_stages);
    gpu_param.global_size[1]  = in_shape->data[1];
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input_attr);
#undef SAFE_FREE_TENSOR_ATTR
    return status;
} /* _topk_initializer() */

DEF_KERNEL_INITIALIZER(_topk_odd_even_sort_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * input_attr   = NULL;
    vsi_size_array_t * in_shape                = NULL;

    VSI_UNREFERENCED(param_size);

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );

    in_shape  = input_attr->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.local_size[0]   = 32;
    gpu_param.local_size[1]   = 1;
    gpu_param.global_size[0]  = 32;
    gpu_param.global_size[1]  = in_shape->data[1];
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input_attr);
#undef SAFE_FREE_TENSOR_ATTR
    return status;
} /* _topk_odd_even_sort_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t num_stages
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _topk_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _topk_kernel_map );
    vx_param_description_t * param_def  = _topk_kernel_param_def;
    vx_kernel_initialize_f  initializer = _topk_initializer;
#define _PACK_SELECT_KEY( in_type, out_type ) \
    ( (in_type) | (out_type << 8) )
    uint32_t key = 0;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    switch (_PACK_SELECT_KEY(in_dtype, out_dtype))
    {
    case _PACK_SELECT_KEY(F32, F32):
    case _PACK_SELECT_KEY(F16, F16):
        key = TOPK_HASH_KEY( F32, F32, num_stages );
        break;
    case _PACK_SELECT_KEY(U32, U32):
    case _PACK_SELECT_KEY(U16, U16):
    case _PACK_SELECT_KEY(U8,  U8):
        key = TOPK_HASH_KEY( U32, U32, num_stages );
        break;
    case _PACK_SELECT_KEY(I32, I32):
    case _PACK_SELECT_KEY(I16, I16):
    case _PACK_SELECT_KEY(I8,  I8):
        key = TOPK_HASH_KEY( I32, I32, num_stages );
        break;
    case _PACK_SELECT_KEY(F32, U32):
    case _PACK_SELECT_KEY(F16, U32):
    case _PACK_SELECT_KEY(F32, U16):
    case _PACK_SELECT_KEY(F16, U16):
    case _PACK_SELECT_KEY(F32, U8):
    case _PACK_SELECT_KEY(F16, U8):
        key = TOPK_HASH_KEY( F32, U32, num_stages );
        break;
    case _PACK_SELECT_KEY(F32, I32):
    case _PACK_SELECT_KEY(F16, I32):
    case _PACK_SELECT_KEY(F32, I16):
    case _PACK_SELECT_KEY(F16, I16):
    case _PACK_SELECT_KEY(F32, I8):
    case _PACK_SELECT_KEY(F16, I8):
        key = TOPK_HASH_KEY( F32, I32, num_stages );
        break;
    default:
        break;
    }
#undef _PACK_SELECT_KEY

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
        kernel->info.numParams   = _cnt_of_array( _topk_kernel_param_def );
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

static vsi_status _query_odd_even_sort_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _topk_odd_even_sort_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _topk_odd_even_sort_kernel_map );
    vx_param_description_t * param_def  = _topk_odd_even_sort_kernel_param_def;
    vx_kernel_initialize_f  initializer = _topk_odd_even_sort_initializer;
#define _PACK_SELECT_KEY( in_type, out_type ) \
    ( (in_type) | (out_type << 8) )
    uint32_t key = 0;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    switch (_PACK_SELECT_KEY(in_dtype, out_dtype))
    {
    case _PACK_SELECT_KEY(F32, F32):
    case _PACK_SELECT_KEY(F16, F16):
        key = TOPK_ODD_EVEN_SORT_HASH_KEY( F32, F32 );
        break;
    case _PACK_SELECT_KEY(U32, U32):
    case _PACK_SELECT_KEY(U16, U16):
    case _PACK_SELECT_KEY(U8,  U8):
        key = TOPK_ODD_EVEN_SORT_HASH_KEY( U32, U32 );
        break;
    case _PACK_SELECT_KEY(I32, I32):
    case _PACK_SELECT_KEY(I16, I16):
    case _PACK_SELECT_KEY(I8,  I8):
        key = TOPK_ODD_EVEN_SORT_HASH_KEY( I32, I32 );
        break;
    case _PACK_SELECT_KEY(F32, U32):
    case _PACK_SELECT_KEY(F16, U32):
    case _PACK_SELECT_KEY(F32, U16):
    case _PACK_SELECT_KEY(F16, U16):
    case _PACK_SELECT_KEY(F32, U8):
    case _PACK_SELECT_KEY(F16, U8):
        key = TOPK_ODD_EVEN_SORT_HASH_KEY2( F32, U32 );
        break;
    case _PACK_SELECT_KEY(F32, I32):
    case _PACK_SELECT_KEY(F16, I32):
    case _PACK_SELECT_KEY(F32, I16):
    case _PACK_SELECT_KEY(F16, I16):
    case _PACK_SELECT_KEY(F32, I8):
    case _PACK_SELECT_KEY(F16, I8):
        key = TOPK_ODD_EVEN_SORT_HASH_KEY2( F32, I32 );
        break;
    default:
        break;
    }
#undef _PACK_SELECT_KEY

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
        kernel->info.numParams   = _cnt_of_array( _topk_odd_even_sort_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_TOPK_ODD_EVEN_SORT_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t block_size = inputs[0]->attr.size[0];
    vsi_size_t block_num = 1;
    uint32_t i = 0;
    vsi_nn_tensor_t* rs_tensors[5] = { NULL };
    vsi_nn_tensor_attr_t attr;
    vsi_size_t shape[2][VSI_NN_MAX_DIM_NUM] = {{ 0 }};
    int32_t width = (int32_t)block_size;
    int32_t top_k = vsi_nn_kernel_param_get_int32(params, "top_k");
    int32_t num_stages = (int32_t)vsi_nn_max(ceil(log10(block_size / 2.0f) / log10(2.0f)), 0);
    vsi_bool is_odd_even_sort = FALSE;
    size_t param_num = _TOPK_PARAM_NUM;
    float inputScale  = vsi_nn_get_tensor_scale(inputs[0]);
    float inputTail   = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float outputScale = vsi_nn_get_tensor_scale(outputs[0]);
    float outputTail  = (float)vsi_nn_get_tensor_zero_point(outputs[0]);

    outputScale = 1.0f / outputScale;
    inputTail   = -(inputTail * inputScale);

    for (i = 1; i < inputs[0]->attr.dim_num; i ++)
    {
        block_num = block_num * inputs[0]->attr.size[i];
    }

    if ((vsi_nn_is_same_type(inputs[0], outputs[0]) == FALSE ||
       outputs[1]->attr.dtype.vx_type != VSI_NN_TYPE_INT32 ) &&
      !(inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16 &&
        (outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8 ||
        outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16)))
    {
        return NULL;
    }

    shape[0][0] = block_size;
    shape[0][1] = block_num;
    shape[1][0] = top_k;
    shape[1][1] = block_num;

    rs_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shape[0], 2 );

    if (num_stages < 7)
    {
        status = _query_kernel( kernel, inputs, outputs, num_stages );

        rs_tensors[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], shape[1], 2 );
        CHECK_PTR_FAIL_GOTO(rs_tensors[1], "Create tensor failed", final);
        rs_tensors[2] = vsi_nn_reshape_tensor( graph,
            outputs[1], shape[1], 2 );
        CHECK_PTR_FAIL_GOTO(rs_tensors[2], "Create tensor failed", final);
    }
    else
    {
        status = _query_odd_even_sort_kernel( kernel, inputs, outputs );
        is_odd_even_sort = TRUE;
        param_num = _TOPK_ODD_EVEN_SORT_PARAM_NUM;

        memcpy( &attr, &(rs_tensors[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
        rs_tensors[1] = vsi_nn_CreateTensor( graph, &attr );
        CHECK_PTR_FAIL_GOTO(rs_tensors[1], "Create tensor failed", final);
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        rs_tensors[2] = vsi_nn_CreateTensor( graph, &attr );
        CHECK_PTR_FAIL_GOTO(rs_tensors[2], "Create tensor failed", final);
        rs_tensors[3] = vsi_nn_reshape_tensor( graph,
            outputs[0], shape[1], 2 );
        CHECK_PTR_FAIL_GOTO(rs_tensors[3], "Create tensor failed", final);
        rs_tensors[4] = vsi_nn_reshape_tensor( graph,
            outputs[1], shape[1], 2 );
        CHECK_PTR_FAIL_GOTO(rs_tensors[4], "Create tensor failed", final);

        input_num = 3;
    }
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = (uint32_t)(input_num + output_num);
            /* Set inputs and outputs  */
            vsi_nn_kernel_node_pack_io( node_params, param_num,
                    rs_tensors, input_num, &rs_tensors[input_num], output_num );
            /* Pass parameters to node. */
            node_params[index++] = vsi_nn_kernel_scalar_create(graph, I32, &inputScale );
            node_params[index++] = vsi_nn_kernel_scalar_create(graph, I32, &inputTail );
            node_params[index++] = vsi_nn_kernel_scalar_create(graph, I32, &outputScale );
            node_params[index++] = vsi_nn_kernel_scalar_create(graph, I32, &outputTail );
            if (is_odd_even_sort)
            {
                node_params[SCALAR_INPUT_SIZE] = vsi_nn_kernel_scalar_create(
                    graph, I32, &width );
            }
            else
            {
                node_params[SCALAR_INPUT_NUM_STAGES] = vsi_nn_kernel_scalar_create(
                    graph, I32, &num_stages );
                node_params[SCALAR_INPUT_WIDTH] = vsi_nn_kernel_scalar_create(
                    graph, I32, &width );
            }

            status  = vsi_nn_kernel_node_pass_param( node, node_params, param_num );
            CHECK_STATUS_FAIL_GOTO( status, final );
        }
    }
final:
    vsi_safe_release_tensor(rs_tensors[0]);
    vsi_safe_release_tensor(rs_tensors[1]);
    vsi_safe_release_tensor(rs_tensors[2]);
    vsi_safe_release_tensor(rs_tensors[3]);
    vsi_safe_release_tensor(rs_tensors[4]);

    if (is_odd_even_sort)
    {
        if (node_params[5])
        {
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
        if (node_params[6])
        {
            vsi_nn_kernel_scalar_release( &node_params[6] );
        }
        if (node_params[7])
        {
            vsi_nn_kernel_scalar_release( &node_params[7] );
        }
        if (node_params[8])
        {
            vsi_nn_kernel_scalar_release( &node_params[8] );
        }
        if (node_params[SCALAR_INPUT_SIZE])
        {
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SIZE] );
        }
    }
    else
    {
        if (node_params[3])
        {
            vsi_nn_kernel_scalar_release( &node_params[3] );
        }
        if (node_params[4])
        {
            vsi_nn_kernel_scalar_release( &node_params[4] );
        }
        if (node_params[5])
        {
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
        if (node_params[6])
        {
            vsi_nn_kernel_scalar_release( &node_params[6] );
        }
        if (node_params[SCALAR_INPUT_NUM_STAGES])
        {
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_NUM_STAGES] );
        }
        if (node_params[SCALAR_INPUT_WIDTH])
        {
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_WIDTH] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( topk, _setup )
