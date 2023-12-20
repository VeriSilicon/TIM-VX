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
    INTERNAL_KERNEL_SCATTER_ELEMENTS,
} _internal_kernel_e;

#define _KERNEL_SOURCE0      "scatter_elements"
#define _KERNEL_SOURCE1      "scatter_elements_add"
#define _KERNEL_SOURCE2      "scatter_elements_mul"

#define STR(a) #a
// Add kernel hashtable here
#define SCATTER_ELEMENTS_HASH_KEY( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS, REDUCTION ) \
        (( IN0_DTYPE ) | ( IN2_DTYPE << 8 ) | ( OUT_DTYPE << 16 ) | ( AXIS << 24 ) | ( REDUCTION << 28 ))

#define PACK_KERNEL_NONE_MAP( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS ) \
        { SCATTER_ELEMENTS_HASH_KEY( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS, VSI_NN_REDUCTION_TYPE_NONE ), \
        CVIVANTE_NAMESPACE("cl.scatter_elements_axis"STR(AXIS)"_"STR(IN0_DTYPE) \
                    "_I32_"STR(IN2_DTYPE)"to"STR(OUT_DTYPE)), \
        _KERNEL_SOURCE0 }

#define PACK_KERNEL_ADD_MAP( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS ) \
        { SCATTER_ELEMENTS_HASH_KEY( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS, VSI_NN_REDUCTION_TYPE_ADD ), \
        CVIVANTE_NAMESPACE("cl.scatter_elements_add_axis"STR(AXIS)"_"STR(IN0_DTYPE) \
                    "_I32_"STR(IN2_DTYPE)"to"STR(OUT_DTYPE)), \
        _KERNEL_SOURCE1 }

#define PACK_KERNEL_MUL_MAP( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS ) \
        { SCATTER_ELEMENTS_HASH_KEY( IN0_DTYPE, IN2_DTYPE, OUT_DTYPE, AXIS, VSI_NN_REDUCTION_TYPE_MUL ), \
        CVIVANTE_NAMESPACE("cl.scatter_elements_mul_axis"STR(AXIS)"_"STR(IN0_DTYPE) \
                    "_I32_"STR(IN2_DTYPE)"to"STR(OUT_DTYPE)), \
        _KERNEL_SOURCE2 }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;


#define PACK_KERNELS_MAP(type) \
    PACK_KERNEL_NONE_MAP( type, type, type, 0 ), \
    PACK_KERNEL_NONE_MAP( type, type, type, 1 ), \
    PACK_KERNEL_ADD_MAP( type, type, type, 0 ), \
    PACK_KERNEL_ADD_MAP( type, type, type, 1 ), \
    PACK_KERNEL_MUL_MAP( type, type, type, 0 ), \
    PACK_KERNEL_MUL_MAP( type, type, type, 1 ), \
    PACK_KERNEL_MUL_MAP( type, type, type, 2 )

static const _kernel_map_type _scatter_elements_kernel_map[] =
{
    // Register kernel here
    PACK_KERNELS_MAP( I8 ),
    PACK_KERNELS_MAP( U8 ),
    PACK_KERNELS_MAP( I16 ),
    PACK_KERNELS_MAP( F16 ),
    PACK_KERNELS_MAP( I32 ),
    PACK_KERNELS_MAP( F32 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _scatter_elements_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _SCATTER_ELEMENTS_PARAM_NUM  _cnt_of_array( _scatter_elements_kernel_param_def )
#define SCALAR_INPUT_AXIS           (4)
#define SCALAR_INPUT_REDUCTION      (5)
#define SCALAR_REF_SCALE            (6)
#define SCALAR_REF_TAIL             (7)
#define SCALAR_UPDATE_SCALE         (8)
#define SCALAR_UPDATE_TAIL          (9)
#define SCALAR_OUTPUT_ZP            (10)
#define SCALAR_INDICES_INNER        (11)
#define SCALAR_INDICES_AXIS         (12)
#define SCALAR_INDICES_OUTER        (13)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_scatter_elements_initializer)
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

    VSI_UNREFERENCED(param_size);

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    out_shape  = output_attr->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.dim = (out_shape->size < 3 || 1 == out_shape->data[2]) ? 2 : 3;
    gpu_param.global_size[0] = out_shape->data[0];
    gpu_param.global_size[1] = out_shape->data[1];
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    return status;
} /* _scatter_elements_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis,
    int32_t reduction
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _scatter_elements_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _scatter_elements_kernel_map );
    vx_param_description_t * param_def  = _scatter_elements_kernel_param_def;
    vx_kernel_initialize_f  initializer = _scatter_elements_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    in2_dtype  = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (in1_dtype != I32)
    {
        return VSI_FAILURE;
    }

    key = SCATTER_ELEMENTS_HASH_KEY( in0_dtype, in2_dtype, out_dtype, axis, reduction );

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
        kernel->info.numParams   = _cnt_of_array( _scatter_elements_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_SCATTER_ELEMENTS_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* reshape_tensors[4] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    int32_t axis = vsi_nn_kernel_param_get_int32(params, "axis");
    int32_t reduction = vsi_nn_kernel_param_get_int32(params, "reduction");
    int32_t new_axis0 = 0;
    int32_t new_axis1 = 0;
    int32_t inner_size = 0;
    int32_t axis_size = 0;
    int32_t outer_size = 0;
    vsi_bool ret = FALSE;
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float output_zp = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float input0_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float input0_tail = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float input2_scale = vsi_nn_get_tensor_scale(inputs[2]);
    float input2_tail = (float)vsi_nn_get_tensor_zero_point(inputs[2]);

#define MAX_SHAPE_SIZE  (0xFFFFFFFF)
    ret = vsi_nn_kernel_optimize_scatter_elements_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num, axis,
            shapes[0], &rank_in, &new_axis0, MAX_SHAPE_SIZE);
    ret &= vsi_nn_kernel_optimize_scatter_elements_shape(
            inputs[1]->attr.size, inputs[1]->attr.dim_num, axis,
            shapes[1], &rank_in, &new_axis1, MAX_SHAPE_SIZE);
#undef MAX_SHAPE_SIZE


    if ( ret && new_axis0 == new_axis1 )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
                inputs[1], shapes[1], rank_in );
        reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
                inputs[2], shapes[1], rank_in );
        reshape_tensors[3] = vsi_nn_reshape_tensor( graph,
                outputs[0], shapes[0], rank_in );

        inner_size = new_axis0 == 0 ? 1 : (int32_t)shapes[1][0];
        axis_size = new_axis0 == 0 ? (int32_t)shapes[1][0] : (int32_t)shapes[1][1];
        outer_size = new_axis0 == 0 ?  (int32_t)shapes[1][1] : rank_in > 2 ? (int32_t)shapes[1][2] : 1;
    }
    else
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, axis, reduction );
    if ( VSI_SUCCESS == status)
    {
        input0_scale = input0_scale / output_scale;
        input0_tail = - input0_tail * input0_scale;
        input2_scale = input2_scale / output_scale;
        input2_tail = - input2_tail * input2_scale;
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SCATTER_ELEMENTS_PARAM_NUM,
                    reshape_tensors, input_num, &reshape_tensors[3], output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(graph, I32, &new_axis0 );
            node_params[SCALAR_INPUT_REDUCTION] = vsi_nn_kernel_scalar_create(graph, I32, &reduction );
            node_params[SCALAR_REF_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &input0_scale );
            node_params[SCALAR_REF_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &input0_tail );
            node_params[SCALAR_UPDATE_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &input2_scale );
            node_params[SCALAR_UPDATE_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &input2_tail );
            node_params[SCALAR_OUTPUT_ZP]   = vsi_nn_kernel_scalar_create(graph, F32, &output_zp );
            node_params[SCALAR_INDICES_INNER]   = vsi_nn_kernel_scalar_create(graph, I32, &inner_size );
            node_params[SCALAR_INDICES_AXIS]   = vsi_nn_kernel_scalar_create(graph, I32, &axis_size );
            node_params[SCALAR_INDICES_OUTER]   = vsi_nn_kernel_scalar_create(graph, I32, &outer_size );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SCATTER_ELEMENTS_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_REDUCTION] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_REF_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_REF_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_UPDATE_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_UPDATE_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
        }
    }

    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );
    vsi_safe_release_tensor( reshape_tensors[2] );
    vsi_safe_release_tensor( reshape_tensors[3] );

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( scatter_elements, _setup )

