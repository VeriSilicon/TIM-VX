/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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

#if !(VX_DEPTH2SPACE_CRD_MODE_SUPPORT)
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

#define _DEPTH2SPACE_CRD_KERNEL_SOURCE           "depth2space_crd"

// Add kernel hashtable here
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_F32TOF32    CVIVANTE_NAMESPACE("cl.depth2space_crd_F32toF32")

// Add kernel hashtable here
#define HASH_DEPTH2SPACE_CRD_KEY(_input0_type, _output_type, _blk_size) \
    ((_input0_type << 24) | (_output_type << 16) | (_blk_size << 8))

#define TENSOR_DEPTH2SPACE_CRD_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_DEPTH2SPACE_CRD_KEY(IN0_TYPE, OUT_TYPE, 0), \
        VX_KERNEL_NAME_DEPTH2SPACE_CRD_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } depth2space_crd_map[] =
{
    TENSOR_DEPTH2SPACE_CRD_KERNELS(F32,  F32,        _DEPTH2SPACE_CRD_KERNEL_SOURCE)
};

/*
 * Kernel params
 */
static vx_param_description_t _depth2space_crd_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _DEPTH2SPACE_CRD_PARAM_NUM  _cnt_of_array( _depth2space_crd_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_depth2space_crd_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    int32_t     output_dims   = 0;
    int32_t     output_width  = 0;
    int32_t     output_height = 0;
    int32_t     output_chn    = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    output_dims = (int32_t)attr[0]->shape->size;
    output_width = (int32_t)(attr[0]->shape->data[0]);
    output_height = (int32_t)(attr[0]->shape->data[1]);
    output_chn = (int32_t)(output_dims > 2 ? attr[0]->shape->data[2] : 1);

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = output_width;
    gpu_param.global_size[1]   = output_height;
    gpu_param.global_size[2]   = output_chn;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _depth2space_crd_initializer() */

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
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_DEPTH2SPACE_CRD_KEY( input0_dtype, output_dtype, 0 );

    for ( i = 0; i < _cnt_of_array(depth2space_crd_map); i ++ )
    {
        if ( depth2space_crd_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(depth2space_crd_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  depth2space_crd_map[i].function_name );
        kernel->info.parameters = _depth2space_crd_kernel_param_def;
        kernel->info.numParams = _DEPTH2SPACE_CRD_PARAM_NUM;
        kernel->info.initialize = _depth2space_crd_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                depth2space_crd_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                depth2space_crd_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_DEPTH2SPACE_CRD_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _DEPTH2SPACE_CRD_PARAM_NUM,
                inputs, 1, outputs, 1 );
            node_params[2] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _DEPTH2SPACE_CRD_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[2] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( depth2space_internal, _setup )
#endif
