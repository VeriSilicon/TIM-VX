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
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

#define _REVERSESEQUENCE_KERNEL_SOURCE_NAME       "reversesequence"

// Add kernel hashtable here
#define REVERSESEQUENCE_HASH_KEY( IN_DTYPE0, IN_DTYPE1, OUT_DTYPE, batch_axis ) \
        (( IN_DTYPE0 << 24 ) | ( IN_DTYPE1 << 16 ) | ( OUT_DTYPE << 8) | (batch_axis) )
#define REVERSESEQUENCE_KERNELS( IN_DTYPE0, IN_DTYPE1, OUT_DTYPE, batch_axis ) \
        { REVERSESEQUENCE_HASH_KEY( IN_DTYPE0, IN_DTYPE1, OUT_DTYPE, batch_axis ), \
        CVIVANTE_NAMESPACE("cl.reversesequence_"#IN_DTYPE0"to"#OUT_DTYPE#batch_axis), \
        _REVERSESEQUENCE_KERNEL_SOURCE_NAME },

typedef enum
{
    _axis1 = 0,
    _axis2
} vsi_nn_kernel_batch_axis_type_e;

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _reversesequence_kernel_map[] =
{
    // Register kernel here
    REVERSESEQUENCE_KERNELS( F32, I32, F32, _axis1)
    REVERSESEQUENCE_KERNELS( F32, I32, U32, _axis1)
    REVERSESEQUENCE_KERNELS( F32, I32, I32, _axis1)
    REVERSESEQUENCE_KERNELS( U32, I32, U32, _axis1)
    REVERSESEQUENCE_KERNELS( U32, I32, F32, _axis1)
    REVERSESEQUENCE_KERNELS( I32, I32, I32, _axis1)
    REVERSESEQUENCE_KERNELS( I32, I32, F32, _axis1)
    REVERSESEQUENCE_KERNELS( BF16, I32, BF16, _axis1)

    REVERSESEQUENCE_KERNELS( F32, I32, F32, _axis2)
    REVERSESEQUENCE_KERNELS( F32, I32, U32, _axis2)
    REVERSESEQUENCE_KERNELS( F32, I32, I32, _axis2)
    REVERSESEQUENCE_KERNELS( U32, I32, U32, _axis2)
    REVERSESEQUENCE_KERNELS( U32, I32, F32, _axis2)
    REVERSESEQUENCE_KERNELS( I32, I32, I32, _axis2)
    REVERSESEQUENCE_KERNELS( I32, I32, F32, _axis2)
    REVERSESEQUENCE_KERNELS( BF16, I32, BF16, _axis2)
};


/*
 * Kernel params
 */
static vx_param_description_t _reversesequence_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _REVERSESEQUENCE_PARAM_NUM  _cnt_of_array( _reversesequence_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_reversesequence_initializer)
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
    vx_tensor  input = (vx_tensor)param[0];
    vsi_nn_kernel_tensor_attr_t *input_attr  = NULL;
    vsi_size_array_t            *input_shape = NULL;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input );
    CHECK_PTR_FAIL_GOTO( input_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    input_shape = input_attr->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = (input_shape->data[0] +  gpu_param.global_scale[0] - 1)
                                        /  gpu_param.global_scale[0];
    gpu_param.global_size[1]   = (input_shape->data[1] +  gpu_param.global_scale[1] - 1)
                                        /  gpu_param.global_scale[1];
    gpu_param.global_size[2]   = (input_shape->data[2] +  gpu_param.global_scale[2] - 1)
                                        /  gpu_param.global_scale[2];
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input_attr);
    }

    return status;
} /* _reversesequence_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t batch_axis
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _reversesequence_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _reversesequence_kernel_map );
    vx_param_description_t * param_def  = _reversesequence_kernel_param_def;
    vx_kernel_initialize_f  initializer = _reversesequence_initializer;
    vsi_nn_kernel_batch_axis_type_e axis_type = _axis1;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (batch_axis == 2)
    {
        axis_type = _axis2;
    }

#define _PACK_SELECT_KEY( in_dtype, out_dtype ) \
    (( in_dtype ) | (out_dtype << 8 ))
    switch(_PACK_SELECT_KEY( in_dtype, out_dtype ))
    {
    case _PACK_SELECT_KEY(F16, F16):
    case _PACK_SELECT_KEY(F32, F32):
        key = REVERSESEQUENCE_HASH_KEY( F32, I32, F32, axis_type);
        break;
    case _PACK_SELECT_KEY(F16, U8):
    case _PACK_SELECT_KEY(F32, U8):
        key = REVERSESEQUENCE_HASH_KEY( F32, I32, U32, axis_type);
        break;
    case _PACK_SELECT_KEY(F16, I8):
    case _PACK_SELECT_KEY(F32, I8):
    case _PACK_SELECT_KEY(F16, I16):
    case _PACK_SELECT_KEY(F32, I16):
        key = REVERSESEQUENCE_HASH_KEY( F32, I32, I32, axis_type);
        break;
    case _PACK_SELECT_KEY(U8, U8):
        key = REVERSESEQUENCE_HASH_KEY( U32, I32, U32, axis_type);
        break;
    case _PACK_SELECT_KEY(U8, F16):
    case _PACK_SELECT_KEY(U8, F32):
        key = REVERSESEQUENCE_HASH_KEY( U32, I32, F32, axis_type);
        break;
    case _PACK_SELECT_KEY(I8, I8):
    case _PACK_SELECT_KEY(I16, I16):
        key = REVERSESEQUENCE_HASH_KEY( I32, I32, I32, axis_type);
        break;
    case _PACK_SELECT_KEY(I8, F16):
    case _PACK_SELECT_KEY(I8, F32):
    case _PACK_SELECT_KEY(I16, F16):
    case _PACK_SELECT_KEY(I16, F32):
        key = REVERSESEQUENCE_HASH_KEY( I32, I32, F32, axis_type);
        break;
    case _PACK_SELECT_KEY(BF16, BF16):
        key = REVERSESEQUENCE_HASH_KEY( BF16, I32, BF16, axis_type);
        break;
    default:
        key = REVERSESEQUENCE_HASH_KEY( in_dtype, I32, out_dtype, axis_type);
        break;
    }

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
        kernel->info.numParams   = _cnt_of_array( _reversesequence_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_REVERSESEQUENCE_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t batch_axis = vsi_nn_kernel_param_get_int32(params, "batch_axis");
    float   outputScale  = vsi_nn_get_tensor_scale(outputs[0]);
    float   outputTail   = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float   inputScale   = vsi_nn_get_tensor_scale(inputs[0]);
    float   inputTail    = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float   inoutScale   = inputScale / outputScale;
    float   inoutTail    = outputTail - inputTail * inoutScale;

    if ( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ))
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, batch_axis );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            uint32_t index = 3;
            vsi_nn_kernel_node_pack_io( node_params, _REVERSESEQUENCE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &inoutScale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &inoutTail );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _REVERSESEQUENCE_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( reversesequence, _setup )

