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
  CVIVANTE_NAMESPACE("evis.gather_elements_axis"STR(AXIS)"_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)), \
  _GATHER_ELEMENTS_KERNEL_SOURCE}

#define PACK_KERNEL_2D_MAP( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
  { GATHER_ELEMENTS_HASH_KEY( AXIS, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1 ), \
  CVIVANTE_NAMESPACE("evis.gather_elements_axis"STR(AXIS)"_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
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
    PACK_KERNEL_3D_MAP( 0, F16, I32, F16 ),
    PACK_KERNEL_3D_MAP( 0, I16, I32, I16 ),
    PACK_KERNEL_3D_MAP( 0, I8,  I32, I8 ),
    PACK_KERNEL_3D_MAP( 0, U8,  I32, U8 ),
    PACK_KERNEL_3D_MAP( 1, F16, I32, F16 ),
    PACK_KERNEL_3D_MAP( 1, I16, I32, I16 ),
    PACK_KERNEL_3D_MAP( 1, I8,  I32, I8 ),
    PACK_KERNEL_3D_MAP( 1, U8,  I32, U8 ),
    PACK_KERNEL_3D_MAP( 2, F16, I32, F16 ),
    PACK_KERNEL_3D_MAP( 2, I16, I32, I16 ),
    PACK_KERNEL_3D_MAP( 2, I8,  I32, I8 ),
    PACK_KERNEL_3D_MAP( 2, U8,  I32, U8 ),

    PACK_KERNEL_2D_MAP( 0, F16, I32, F16 ),
    PACK_KERNEL_2D_MAP( 0, I16, I32, I16 ),
    PACK_KERNEL_2D_MAP( 0, I8,  I32, I8 ),
    PACK_KERNEL_2D_MAP( 0, U8,  I32, U8 ),
    PACK_KERNEL_2D_MAP( 1, F16, I32, F16 ),
    PACK_KERNEL_2D_MAP( 1, I16, I32, I16 ),
    PACK_KERNEL_2D_MAP( 1, I8,  I32, I8 ),
    PACK_KERNEL_2D_MAP( 1, U8,  I32, U8 ),
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
};
#define _GATHER_ELEMENTS_PARAM_NUM  _cnt_of_array( _gather_elements_kernel_param_def )
#define SCALAR_INPUT_AXIS        (3)

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
        {1, 1, 1},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * input_attr  = NULL;
    vsi_nn_kernel_tensor_attr_t * output_attr = NULL;
    vsi_size_array_t * out_shape              = NULL;
    int32_t axis = 0;
    int32_t axis_size = 0;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_INPUT_AXIS], &axis);

    out_shape = output_attr->shape;
    axis_size = (int32_t)input_attr->shape->data[axis];
    if (axis == 0)
    {
        gpu_param.global_scale[0] = 4;
    }

    gpu_param.dim = (out_shape->size < 3 || 1 == out_shape->data[2]) ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    status  = vsi_nn_kernel_gpu_config( node, &gpu_param );
    status |= vsi_nn_kernel_gpu_add_param( node, "axis_size", &axis_size );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input_attr);
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

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (in0_dtype == BF16)
    {
        in0_dtype = F16;
    }
    if (out_dtype == BF16)
    {
        out_dtype = F16;
    }

    key = GATHER_ELEMENTS_HASH_KEY( axis, in0_dtype, in1_dtype, out_dtype, img_2d );

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
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
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
    int32_t axis = vsi_nn_kernel_param_get_int32(params, "axis");

    if ( vsi_nn_is_same_type(inputs[0], outputs[0]) == FALSE )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, axis );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GATHER_ELEMENTS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_INPUT_AXIS]  = vsi_nn_kernel_scalar_create( graph, I32, &axis );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GATHER_ELEMENTS_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( gather_elements, _setup )
