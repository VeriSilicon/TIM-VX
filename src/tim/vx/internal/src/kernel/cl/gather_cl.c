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

#define _GATHER_KERNEL_SOURCE      "gather"

// Add kernel hashtable here
#define VX_KERNEL_NAME_GATHER_U8TOU8       CVIVANTE_NAMESPACE("cl.gather_U8toU8")
#define VX_KERNEL_NAME_GATHER_F16TOF16     CVIVANTE_NAMESPACE("cl.gather_F16toF16")
#define VX_KERNEL_NAME_GATHER_I32TOI32     CVIVANTE_NAMESPACE("cl.gather_I32toI32")
#define VX_KERNEL_NAME_GATHER_F32TOF32     CVIVANTE_NAMESPACE("cl.gather_F32toF32")

// Add kernel hashtable here
#define HASH_GATHER_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define TENSOR_GATHER_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 0), \
        VX_KERNEL_NAME_GATHER_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } gather_map[] =
{
    TENSOR_GATHER_KERNELS(U8,  I32, U8,       _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_KERNELS(F16, I32, F16,       _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_KERNELS(I32, I32, I32,       _GATHER_KERNEL_SOURCE)
    TENSOR_GATHER_KERNELS(F32, I32, F32,       _GATHER_KERNEL_SOURCE)
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
    // Add kererl parameters here
};
#define _GATHER_PARAM_NUM  _cnt_of_array( _gather_kernel_param_def )

static vsi_status cal_gather_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t idxFlg
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    vsi_size_t elementCnt = 1;
#define VSI_NN_MAX_IMAGE_WIDTH  (65536)

    for(i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for(i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    if(idxFlg && elementCnt < VSI_NN_MAX_IMAGE_WIDTH)
    {
        sizes[0] = elementCnt;
        sizes[1] = 1;
        status = VSI_SUCCESS;
    }
    else
    {
        if((elementCnt / block_size) < VSI_NN_MAX_IMAGE_WIDTH)
        {
            sizes[0] = block_size;
            sizes[1] = elementCnt / block_size;
            status = VSI_SUCCESS;
        }
    }
#undef VSI_NN_MAX_IMAGE_WIDTH

    return status;
} /* _get_EltOP_tensor_reshape_size */

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
    vsi_ssize_t       indices_num = 1;
    size_t      input_dims1 = 0;
    size_t     i           = 0;

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
    for (i = 0; i < input_dims1; i++)
    {
        indices_num *= input1_shape->data[i];
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((block_size + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
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
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_GATHER_KEY( input0_dtype, I32, output_dtype, 0 );

    for( i = 0; i < _cnt_of_array(gather_map); i ++ )
    {
        if( gather_map[i].key == key )
        {
            break;
        }
    }

    if( i < _cnt_of_array(gather_map) )
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
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t block_num   = vsi_nn_kernel_param_get_int32( params, "block_num" );
    int32_t axis_num    = vsi_nn_kernel_param_get_int32( params, "axis_num" );
    int32_t indices_num = vsi_nn_kernel_param_get_int32( params, "indices_num" );

    status = cal_gather_tensor_reshape_size(&inputs[0], shapes[0], block_size, 0);
    status |= cal_gather_tensor_reshape_size(&inputs[1], shapes[1], 1, 1);
    status |= cal_gather_tensor_reshape_size(&outputs[0], shapes[2], block_size, 0);
    if(status != VSI_SUCCESS)
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 0;
#define RESHAPE_DIM 2
            /* Pass parameters to node. */
            node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[0], RESHAPE_DIM );
            node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[1], RESHAPE_DIM );
            node_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], RESHAPE_DIM );
#undef RESHAPE_DIM
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &indices_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GATHER_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_tensor_release( &node_params[0] );
            vsi_nn_kernel_tensor_release( &node_params[1] );
            vsi_nn_kernel_tensor_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( gather, _setup )

