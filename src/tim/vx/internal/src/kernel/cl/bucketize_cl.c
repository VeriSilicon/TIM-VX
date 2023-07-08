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
typedef enum
{
    INTERNAL_KERNEL_BUCKETIZE,
} _internal_kernel_e;

#define STR(a) #a

// Add kernel hashtable here
#define BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        (( IN0_DTYPE ) | ( IN1_DTYPE << 8 ) | ( OUT_DTYPE << 16 ) | (RIGHT << 24) | (IMG_2D << 25))

#define PACK_KERNEL_3D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        { BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ), \
        CVIVANTE_NAMESPACE("cl.bucketize_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)), \
        "bucketize" }
#define PACK_KERNEL_2D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        { BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ), \
        CVIVANTE_NAMESPACE("cl.bucketize_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        "bucketize" }
#define PACK_KERNEL_RIGHT_3D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        { BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ), \
        CVIVANTE_NAMESPACE("cl.bucketize_right_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)), \
        "bucketize" }
#define PACK_KERNEL_RIGHT_2D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        { BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ), \
        CVIVANTE_NAMESPACE("cl.bucketize_right_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        "bucketize" }

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
    PACK_KERNEL_3D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0, 0 ), \
    PACK_KERNEL_2D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0, 1 ), \
    PACK_KERNEL_RIGHT_3D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1, 0 ), \
    PACK_KERNEL_RIGHT_2D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1, 1 ),

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _bucketize_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32,  F32,  I32 )
    PACK_KERNEL_MAP( I32,  I32,  I32 )
    PACK_KERNEL_MAP( U32,  U32,  I32 )
    PACK_KERNEL_MAP( BF16, BF16, I32 )
};


/*
 * Kernel params
 */
static vx_param_description_t _bucketize_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _BUCKETIZE_PARAM_NUM  _cnt_of_array( _bucketize_kernel_param_def )
#define SCALAR_BOUNDARIES_VALUE         (3)
#define SCALAR_SCALE0_VALUE             (4)
#define SCALAR_TAIL0_VALUE              (5)
#define SCALAR_SCALE1_VALUE             (6)
#define SCALAR_TAIL1_VALUE              (7)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_bucketize_initializer)
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
    vsi_size_array_t * out_shape                = NULL;

    VSI_UNREFERENCED(param_size);

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
#undef SAFE_FREE_TENSOR_ATTR
    return status;
} /* _bucketize_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t right,
    vsi_bool is_img2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _bucketize_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _bucketize_kernel_map );
    vx_param_description_t * param_def  = _bucketize_kernel_param_def;
    vx_kernel_initialize_f  initializer = _bucketize_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

#define _PACK_SELECT_KEY( in0_dtype, in1_dtype ) \
    ( ( in0_dtype ) | ( in1_dtype << 8 ))

    switch (_PACK_SELECT_KEY(in0_dtype, in1_dtype))
    {
    case _PACK_SELECT_KEY(F32, F32):
    case _PACK_SELECT_KEY(F16, F16):
        key = BUCKETIZE_HASH_KEY( F32, F32, out_dtype, right, is_img2d );
        break;
    case _PACK_SELECT_KEY(I8,  I8):
    case _PACK_SELECT_KEY(I16, I16):
    case _PACK_SELECT_KEY(I32, I32):
        key = BUCKETIZE_HASH_KEY( I32, I32, out_dtype, right, is_img2d );
        break;
    case _PACK_SELECT_KEY(U8,  U8):
    case _PACK_SELECT_KEY(U16, U16):
    case _PACK_SELECT_KEY(U32, U32):
        key = BUCKETIZE_HASH_KEY( U32, U32, out_dtype, right, is_img2d );
        break;
    default:
        key = BUCKETIZE_HASH_KEY( in0_dtype, in1_dtype, out_dtype, right, is_img2d );
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
        kernel->info.numParams   = _cnt_of_array( _bucketize_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_BUCKETIZE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    float input0_scale= vsi_nn_get_tensor_scale(inputs[0]);
    float input0_tail = -input0_scale * (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float input1_scale= vsi_nn_get_tensor_scale(inputs[1]);
    float input1_tail = -input0_scale * (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    int32_t boundaries_size = (int32_t)inputs[1]->attr.size[0];
    vsi_bool image_2d = FALSE;
    int32_t right = vsi_nn_kernel_param_get_int32( params, "right" );

    if( !vsi_nn_kernel_gpu_check_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num ) ||
        boundaries_size >= GPU_TENSOR_MAX_WIDTH )
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);

    status = _query_kernel( kernel, inputs, outputs, right, image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BUCKETIZE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_BOUNDARIES_VALUE] = vsi_nn_kernel_scalar_create( graph, I32, &boundaries_size );
            node_params[SCALAR_SCALE0_VALUE]  = vsi_nn_kernel_scalar_create( graph, F32, &input0_scale );
            node_params[SCALAR_TAIL0_VALUE]   = vsi_nn_kernel_scalar_create(graph, F32, &input0_tail );
            node_params[SCALAR_SCALE1_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &input1_scale );
            node_params[SCALAR_TAIL1_VALUE]  = vsi_nn_kernel_scalar_create(graph, F32, &input1_tail );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _BUCKETIZE_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_BOUNDARIES_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE0_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_TAIL0_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE1_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_TAIL1_VALUE] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( bucketize, _setup )

