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
#define KERNEL_SOURCE_1    "scatter_nd"

 typedef enum
{
    _1D = 0,
    _2D,
    _3D
} vsi_nn_kernel_coord_type_e;

#define HASH_SCATTER_ND_KEY(_input0_type, _input1_type, _output_type, _coord_dim) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_coord_dim))

#define HASH_SCATTER_ND_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE, COORD_TYPE) \
    CVIVANTE_NAMESPACE("cl.scatter_nd_"#SRC0_TYPE"to"#DST_TYPE#COORD_TYPE)

#define TENSOR_SCATTER_ND_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, COORD_TYPE, SOURCE) \
    { HASH_SCATTER_ND_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, COORD_TYPE), \
        HASH_SCATTER_ND_SH_KERNEL_NAME(IN1_TYPE, OUT_TYPE, COORD_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } scatter_nd_map[] =
{
    TENSOR_SCATTER_ND_KERNELS(I32, I32, I32,  _1D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, I32, I32,  _2D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, I32, I32,  _3D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, U32, U32,  _1D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, U32, U32,  _2D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, U32, U32,  _3D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, F32, F32,  _1D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, F32, F32,  _2D,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, F32, F32,  _3D,      KERNEL_SOURCE_1)
};

/*
 * Kernel params
 */
static vx_param_description_t _scatter_nd_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _SCATTER_ND_PARAM_NUM          _cnt_of_array(_scatter_nd_kernel_param_def)

static vsi_status cal_scatter_nd_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t coordDim,
    vsi_size_t* width,
    vsi_size_t* area,
    int32_t* newDim
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    vsi_size_t elementCnt = 1;

    if(coordDim != 0 && (width == NULL || area == NULL))
    {
        return status;
    }

#define VSI_NN_MAX_IMAGE_WIDTH  GPU_TENSOR_MAX_WIDTH

    newDim[0] = 0;
    for(i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for(i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    if((elementCnt / block_size) < VSI_NN_MAX_IMAGE_WIDTH)
    {
        sizes[0] = block_size;
        sizes[1] = elementCnt / block_size;
        status = VSI_SUCCESS;
        newDim[0] = 2;
    }
    else
    {
        return status;
    }

    if(coordDim == 1) // index shape
    {
        *width = 0;
        *area = 0;
    }
    else if(coordDim == 2)
    {
        *width = input_size[dims_num - 2];
        *area = 0;
    }
    else if(coordDim == 3)
    {
        *width = input_size[dims_num - 3];
        *area = input_size[dims_num - 3] * input_size[dims_num - 2];
    }
#undef VSI_NN_MAX_IMAGE_WIDTH

    return status;
} /* _get_EltOP_tensor_reshape_size */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_scatter_nd_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
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
    vsi_ssize_t       block_size  = 0;
    vsi_ssize_t       height = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    block_size = attr[0]->shape->data[0];
    height = attr[0]->shape->data[1];

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _scatter_nd_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t coord_dim
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input1_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_nn_kernel_coord_type_e coord_type = _1D;
    uint32_t key = 0;
    int i = 0;

    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    if(coord_dim == 1)
    {
        coord_type = _1D;
    }
    else if(coord_dim == 2)
    {
        coord_type = _2D;
    }
    else if(coord_dim == 3)
    {
        coord_type = _3D;
    }

    key = HASH_SCATTER_ND_KEY( I32, input1_dtype, output_dtype, coord_type );

    for( i = 0; i < _cnt_of_array(scatter_nd_map); i ++ )
    {
        if( scatter_nd_map[i].key == key )
        {
            break;
        }
    }

    if( i < _cnt_of_array(scatter_nd_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  scatter_nd_map[i].function_name );
        kernel->info.parameters = _scatter_nd_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _scatter_nd_kernel_param_def );
        kernel->info.initialize = _scatter_nd_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                scatter_nd_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_SCATTER_ND_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t coord_dim  = vsi_nn_kernel_param_get_int32( params, "coord_dim" );
    int32_t idx_num  = vsi_nn_kernel_param_get_int32( params, "idx_num" );
    int32_t rs_in_dim = 0, rs_idx_dim = 0, rs_out_dim = 0;
    vsi_size_t width = 0, area = 0;

    status = cal_scatter_nd_tensor_reshape_size(&inputs[0], shapes[0], coord_dim, 0, NULL, NULL, &rs_in_dim);
    status |= cal_scatter_nd_tensor_reshape_size(&inputs[1], shapes[1], block_size, 0, NULL, NULL, &rs_idx_dim);
    status |= cal_scatter_nd_tensor_reshape_size(&outputs[0], shapes[2], block_size, coord_dim,
                                            &width, &area, &rs_out_dim);
    if(status != VSI_SUCCESS)
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, coord_dim );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 0;
            /* Pass parameters to node. */
            node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[0], rs_in_dim );
            node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[1], rs_idx_dim );
            node_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &area );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &idx_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SCATTER_ND_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_tensor_release( &node_params[0] );
            vsi_nn_kernel_tensor_release( &node_params[1] );
            vsi_nn_kernel_tensor_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( scatter_nd, _setup )
