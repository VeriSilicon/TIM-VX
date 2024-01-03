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

#if !(VX_TENSOR_GATHER_API_SUPPORT)
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
typedef enum
{
    INTERNAL_KERNEL_GATHER,
} _internal_kernel_e;

#define _GATHER_KERNEL_SOURCE           "gather"
#define _GATHER_BATCH_KERNEL_SOURCE     "gather_batch"
#define _GATHER_ARRAY_KERNEL_SOURCE     "gather_array"

// Add kernel hashtable here
#define VX_KERNEL_NAME_GATHER_U8TOU8       CVIVANTE_NAMESPACE("cl.gather_U8toU8")
#define VX_KERNEL_NAME_GATHER_F16TOF16     CVIVANTE_NAMESPACE("cl.gather_F16toF16")
#define VX_KERNEL_NAME_GATHER_I32TOI32     CVIVANTE_NAMESPACE("cl.gather_I32toI32")
#define VX_KERNEL_NAME_GATHER_F32TOF32     CVIVANTE_NAMESPACE("cl.gather_F32toF32")

#define VX_KERNEL_NAME_GATHER_BATCH_U8TOU8       CVIVANTE_NAMESPACE("cl.gather_batch_U8toU8")
#define VX_KERNEL_NAME_GATHER_BATCH_F16TOF16     CVIVANTE_NAMESPACE("cl.gather_batch_F16toF16")
#define VX_KERNEL_NAME_GATHER_BATCH_I32TOI32     CVIVANTE_NAMESPACE("cl.gather_batch_I32toI32")
#define VX_KERNEL_NAME_GATHER_BATCH_F32TOF32     CVIVANTE_NAMESPACE("cl.gather_batch_F32toF32")

#define VX_KERNEL_NAME_GATHER_ARRAY_U8TOU8       CVIVANTE_NAMESPACE("cl.gather_array_U8toU8")
#define VX_KERNEL_NAME_GATHER_ARRAY_F16TOF16     CVIVANTE_NAMESPACE("cl.gather_array_F16toF16")
#define VX_KERNEL_NAME_GATHER_ARRAY_I32TOI32     CVIVANTE_NAMESPACE("cl.gather_array_I32toI32")
#define VX_KERNEL_NAME_GATHER_ARRAY_F32TOF32     CVIVANTE_NAMESPACE("cl.gather_array_F32toF32")

// Add kernel hashtable here
#define HASH_GATHER_KEY(_input0_type, _input1_type, _output_type, _is_array, _batch) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_is_array << 4) | (_batch))

#define TENSOR_GATHER_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 0, 0), \
        VX_KERNEL_NAME_GATHER_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_GATHER_BATCH_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 0, 1), \
        VX_KERNEL_NAME_GATHER_BATCH_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_GATHER_ARRAY_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 1, 0), \
        VX_KERNEL_NAME_GATHER_ARRAY_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } gather_map[] =
{
    TENSOR_GATHER_KERNELS(U8,  I32, U8,        _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_KERNELS(F16, I32, F16,       _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_KERNELS(I32, I32, I32,       _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_KERNELS(F32, I32, F32,       _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_BATCH_KERNELS(U8,  I32, U8,  _GATHER_BATCH_KERNEL_SOURCE)
    TENSOR_GATHER_BATCH_KERNELS(F16, I32, F16, _GATHER_BATCH_KERNEL_SOURCE)
    TENSOR_GATHER_BATCH_KERNELS(I32, I32, I32, _GATHER_BATCH_KERNEL_SOURCE)
    TENSOR_GATHER_BATCH_KERNELS(F32, I32, F32, _GATHER_BATCH_KERNEL_SOURCE)
    TENSOR_GATHER_ARRAY_KERNELS(U8,  I32, U8,  _GATHER_ARRAY_KERNEL_SOURCE)
    TENSOR_GATHER_ARRAY_KERNELS(F16, I32, F16, _GATHER_ARRAY_KERNEL_SOURCE)
    TENSOR_GATHER_ARRAY_KERNELS(I32, I32, I32, _GATHER_ARRAY_KERNEL_SOURCE)
    TENSOR_GATHER_ARRAY_KERNELS(F32, I32, F32, _GATHER_ARRAY_KERNEL_SOURCE)
};

/*
 * Kernel params
 */
static vx_param_description_t _gather_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GATHER_PARAM_NUM  _cnt_of_array( _gather_kernel_param_def )

static vsi_status cal_gather_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    vsi_size_t batch_dims,
    uint32_t idxFlg,
    int32_t* arrayFlg
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    vsi_size_t elementCnt = 1;
    vsi_size_t outerCnt = 1;
#define VSI_NN_MAX_IMAGE_WIDTH  GPU_TENSOR_MAX_WIDTH

    for (i = 0; i < dims_num - batch_dims; ++i)
    {
        elementCnt *= input_size[i];
    }

    for (; i < dims_num; ++i)
    {
        outerCnt *= input_size[i];
    }

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    if (idxFlg && elementCnt < VSI_NN_MAX_IMAGE_WIDTH)
    {
        sizes[0] = elementCnt;
        sizes[1] = outerCnt;
        status = VSI_SUCCESS;
    }
    else
    {
        sizes[0] = block_size;
        sizes[1] = elementCnt / block_size;
        sizes[2] = outerCnt;
        if ((elementCnt / block_size) >= VSI_NN_MAX_IMAGE_WIDTH)
        {
            arrayFlg[0] |= 1;
        }
        status = VSI_SUCCESS;
    }
#undef VSI_NN_MAX_IMAGE_WIDTH

    return status;
} /* cal_gather_tensor_reshape_size */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_gather_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * input1_shape = NULL;
    int32_t       block_size  = 0;
    int32_t       block_num   = 0;
    vsi_ssize_t   indices_num = 1;
    size_t        input_dims1 = 0;
    size_t        i           = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &block_size);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &block_num);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input1_shape  = attr[1]->shape;
    input_dims1   = input1_shape->size;
    for (i = 0; i < input_dims1 - 1; i++)
    {
        indices_num *= input1_shape->data[i];
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = indices_num;
    gpu_param.global_size[2]   = block_num;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    return status;
} /* _gather_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t is_batch,
    int32_t is_array
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == I8)
    {
        input0_dtype = I32;
    }

    if (output_dtype == I8)
    {
        output_dtype = I32;
    }

    key = HASH_GATHER_KEY( input0_dtype, I32, output_dtype, is_array, is_batch );

    for ( i = 0; i < _cnt_of_array(gather_map); i ++ )
    {
        if ( gather_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(gather_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  gather_map[i].function_name );
        kernel->info.parameters = _gather_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _gather_kernel_param_def );
        kernel->info.initialize = _gather_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                gather_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                gather_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_GATHER_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    int32_t axis        = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t batch_dims  = vsi_nn_kernel_param_get_int32( params, "batch_dims" );
    int32_t block_size  = 1;
    int32_t block_num   = 1;
    int32_t axis_num    = 0;
    int32_t indices_num = 1;
    int32_t is_batch    = batch_dims > 0 ? 1 : 0;
    vsi_size_t rs_dim   = batch_dims == 0 ? 2 : 3;
    int32_t is_array    = 0;
    uint32_t i          = 0;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t r_rank = vsi_nn_GetTensorIsScalar(inputs[0]) ? 0 : inputs[0]->attr.dim_num;
    uint32_t q_rank = vsi_nn_GetTensorIsScalar(inputs[1]) ? 0 : inputs[1]->attr.dim_num;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    for (i = 0; i < (uint32_t)axis; ++i)
    {
        block_size *= (int32_t)input_size[i];
    }

    axis_num = (int32_t)input_size[axis];
    for (i = axis + 1; i < r_rank - batch_dims; ++i)
    {
        block_num *= (int32_t)input_size[i];
    }
    for (i = 0; i < q_rank - batch_dims; ++i)
    {
        indices_num *= (int32_t)inputs[1]->attr.size[i];
    }

    is_array = block_size >= GPU_TENSOR_MAX_WIDTH ? 1 : 0;

    status = cal_gather_tensor_reshape_size(&inputs[0], shapes[0], block_size, batch_dims, 0, &is_array);
    status |= cal_gather_tensor_reshape_size(&inputs[1], shapes[1], 1, batch_dims, 1, &is_array);
    status |= cal_gather_tensor_reshape_size(&outputs[0], shapes[2], block_size, batch_dims, 0, &is_array);
    if (status != VSI_SUCCESS)
    {
        return NULL;
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], rs_dim );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        inputs[1], shapes[1], 2 );
    reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[2], rs_dim );

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, is_batch, is_array );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 3;
            int32_t batch = (int32_t)shapes[1][1];

            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _GATHER_PARAM_NUM,
                reshape_tensors, 2, &reshape_tensors[2], 1 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &indices_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &batch );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GATHER_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
        }
    }

    for (i = 0; i < 3; i++)
    {
        vsi_safe_release_tensor(reshape_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( gather, _setup )
#endif
