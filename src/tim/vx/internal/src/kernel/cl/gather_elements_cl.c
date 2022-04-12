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

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_GATHER_ELEMENTS,
} _internal_kernel_e;

#define _GATHER_ELEMENTS_KERNEL_SOURCE      "gather_elements"

#define STR(a) #a
// Add kernel hashtable here
#define GATHER_ELEMENTS_HASH_KEY( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, IMG_2D ) \
        (( AXIS ) | ( IN0_DTYPE << 2 ) | ( IN1_DTYPE << 10 ) | ( OUT_DTYPE << 18 ) | ( IMG_2D << 26 ))
#define PACK_KERNEL_3D_MAP( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
    { GATHER_ELEMENTS_HASH_KEY( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0 ), \
    CVIVANTE_NAMESPACE("cl.gather_elements_axis"STR(AXIS)"_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)), \
    _GATHER_ELEMENTS_KERNEL_SOURCE}

#define PACK_KERNEL_2D_MAP( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
    { GATHER_ELEMENTS_HASH_KEY( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1 ), \
    CVIVANTE_NAMESPACE("cl.gather_elements_axis"STR(AXIS)"_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
    _GATHER_ELEMENTS_KERNEL_SOURCE}

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _gather_elements_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_3D_MAP( 0, F32, I32, F32 ),
    PACK_KERNEL_3D_MAP( 0, I32, I32, I32 ),
    PACK_KERNEL_3D_MAP( 0, U32, I32, U32 ),
    PACK_KERNEL_3D_MAP( 1, F32, I32, F32 ),
    PACK_KERNEL_3D_MAP( 1, I32, I32, I32 ),
    PACK_KERNEL_3D_MAP( 1, U32, I32, U32 ),
    PACK_KERNEL_3D_MAP( 2, F32, I32, F32 ),
    PACK_KERNEL_3D_MAP( 2, I32, I32, I32 ),
    PACK_KERNEL_3D_MAP( 2, U32, I32, U32 ),

    PACK_KERNEL_2D_MAP( 0, F32, I32, F32 ),
    PACK_KERNEL_2D_MAP( 0, I32, I32, I32 ),
    PACK_KERNEL_2D_MAP( 0, U32, I32, U32 ),
    PACK_KERNEL_2D_MAP( 1, F32, I32, F32 ),
    PACK_KERNEL_2D_MAP( 1, I32, I32, I32 ),
    PACK_KERNEL_2D_MAP( 1, U32, I32, U32 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _gather_elements_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GATHER_ELEMENTS_PARAM_NUM  _cnt_of_array( _gather_elements_kernel_param_def )
#define SCALAR_INPUT_SCALE        (3)
#define SCALAR_INPUT_TAIL         (4)
#define SCALAR_INPUT_AXIS_SIZE    (5)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_gather_elements_initializer)
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
    vsi_nn_kernel_tensor_attr_t * output_attr = NULL;
    vsi_size_array_t * out_shape              = NULL;

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
} /* _gather_elements_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _gather_elements_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _gather_elements_kernel_map );
    vx_param_description_t * param_def  = _gather_elements_kernel_param_def;
    vx_kernel_initialize_f  initializer = _gather_elements_initializer;
    int32_t img_2d = (outputs[0]->attr.dim_num < 3 || outputs[0]->attr.size[2] == 1) ? 1 : 0;
    uint32_t key = 0;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

#define _PACK_SELECT_KEY( in0_type, out_type ) \
    ( ( in0_type ) | ( out_type << 8 ))

    switch (_PACK_SELECT_KEY(in0_dtype, out_dtype))
    {
    case _PACK_SELECT_KEY(F32, F32):
    case _PACK_SELECT_KEY(F16, F16):
        key = GATHER_ELEMENTS_HASH_KEY( axis, F32, in1_dtype, F32, img_2d );
        break;
    case _PACK_SELECT_KEY(U32, U32):
    case _PACK_SELECT_KEY(U16, U16):
    case _PACK_SELECT_KEY(U8,  U8):
        key = GATHER_ELEMENTS_HASH_KEY( axis, U32, in1_dtype, U32, img_2d );
        break;
    case _PACK_SELECT_KEY(I32, I32):
    case _PACK_SELECT_KEY(I16, I16):
    case _PACK_SELECT_KEY(I8,  I8):
        key = GATHER_ELEMENTS_HASH_KEY( axis, I32, in1_dtype, I32, img_2d );
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
        kernel->info.numParams   = _cnt_of_array( _gather_elements_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_GATHER_ELEMENTS_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float output_zp = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float input_tail = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    int32_t axis = vsi_nn_kernel_param_get_int32(params, "axis");
    int32_t axis_size = (int32_t)inputs[0]->attr.size[axis];

    status = _query_kernel( kernel, inputs, outputs, axis );
    if ( VSI_SUCCESS == status)
    {
        input_scale = input_scale / output_scale;
        input_tail = output_zp - input_tail * input_scale;
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GATHER_ELEMENTS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_INPUT_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
            node_params[SCALAR_INPUT_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &input_tail );
            node_params[SCALAR_INPUT_AXIS_SIZE] = vsi_nn_kernel_scalar_create(graph, I32, &axis_size );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GATHER_ELEMENTS_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS_SIZE] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( gather_elements, _setup )
