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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS


/*
 * Define kernel meta.
 */
#define HASH_TILE_AXIS0_KEY(_input_type, _output_type, _image_2d) \
    ((_input_type << 12) | (_output_type << 4) | (_image_2d))

#define KERNEL_SOURCE    "tile",

#define HASH_TILE_AXIS0_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.tile_"#SRC0_TYPE"to"#DST_TYPE)

#define TENSOR_TILE_AXIS0_KERNELS(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 0), \
        HASH_TILE_AXIS0_SH_KERNEL_NAME(SRC0_TYPE, OUT_TYPE), \
        KERNEL_SOURCE },

#define TENSOR_TILE_AXIS0_FLOAT(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 0), \
        HASH_TILE_AXIS0_SH_KERNEL_NAME(F32, F32), \
        KERNEL_SOURCE },

#define HASH_TILE_AXIS0_SH_KERNEL_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.tile_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_TILE_AXIS0_KERNELS_2D(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 1), \
        HASH_TILE_AXIS0_SH_KERNEL_2D_NAME(SRC0_TYPE, OUT_TYPE), \
        KERNEL_SOURCE },

#define TENSOR_TILE_AXIS0_FLOAT_2D(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 1), \
        HASH_TILE_AXIS0_SH_KERNEL_2D_NAME(F32, F32), \
        KERNEL_SOURCE },

#define TENSOR_TILE_AXIS0_INT32(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 0), \
        HASH_TILE_AXIS0_SH_KERNEL_NAME(I32, I32), \
        KERNEL_SOURCE },

#define TENSOR_TILE_AXIS0_UINT32(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 0), \
        HASH_TILE_AXIS0_SH_KERNEL_NAME(U32, U32), \
        KERNEL_SOURCE },

#define TENSOR_TILE_AXIS0_INT32_2D(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 1), \
        HASH_TILE_AXIS0_SH_KERNEL_2D_NAME(I32, I32), \
        KERNEL_SOURCE },

#define TENSOR_TILE_AXIS0_UINT32_2D(SRC0_TYPE, OUT_TYPE) \
    {   HASH_TILE_AXIS0_KEY(SRC0_TYPE, OUT_TYPE, 1), \
        HASH_TILE_AXIS0_SH_KERNEL_2D_NAME(U32, U32), \
        KERNEL_SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_TILE_AXIS0_INT32(I8,   I8)
    TENSOR_TILE_AXIS0_INT32(I16,  I16)
    TENSOR_TILE_AXIS0_INT32(I32,  I32)
    TENSOR_TILE_AXIS0_UINT32(U8,  U8)
    TENSOR_TILE_AXIS0_UINT32(U32, U32)
    TENSOR_TILE_AXIS0_FLOAT(F16,  F16)
    TENSOR_TILE_AXIS0_FLOAT(F32,  F32)

    TENSOR_TILE_AXIS0_INT32_2D(I8,   I8)
    TENSOR_TILE_AXIS0_INT32_2D(I16,  I16)
    TENSOR_TILE_AXIS0_INT32_2D(I32,  I32)
    TENSOR_TILE_AXIS0_UINT32_2D(U8,  U8)
    TENSOR_TILE_AXIS0_UINT32_2D(U32, U32)
    TENSOR_TILE_AXIS0_FLOAT_2D(F16,  F16)
    TENSOR_TILE_AXIS0_FLOAT_2D(F32,  F32)
};

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)
#define SCALAR_INPUT_BATCH_IN       (2)
#define SCALAR_INPUT_DEPTH_IN       (3)
#define SCALAR_INPUT_DEPTH_OUT      (4)
#define SCALAR_INPUT_MULTIPLES_0    (5)
#define SCALAR_INPUT_MULTIPLES_1    (6)
#define SCALAR_INPUT_MULTIPLES_2    (7)
#define SCALAR_INPUT_MULTIPLES_3    (8)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_tile_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    gpu_param_t gpu_param = {
        3,         // workdim
        {0, 0, 0}, // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0}, // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0}, // localWorkSize: local group size in thread
        {0, 0, 0}  // globalWorkSize: image size in thread
        };

    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * in_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    in_shape  = attr[0]->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = in_shape->data[0];
    gpu_param.global_size[1] = in_shape->data[1];
    gpu_param.global_size[2] = in_shape->size > 2 ? in_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
    }

    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
    }

    return status;
} /* _tile_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_TILE_AXIS0_KEY( input_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _tile_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_bool _is_supported_axis(vsi_size_t* multiples, vsi_size_t multiples_num)
{
    vsi_size_t i = 0;

    if ( multiples_num < 4)
    {
        return TRUE;
    }
    else if ( multiples_num > 4)
    {
        return FALSE;
    }

    for ( i = 3;  i < multiples_num;  i++)
    {
        if (multiples[i] > 1)
        {
            return FALSE;
        }
    }

    return TRUE;
}

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
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t i = 0;
    vsi_size_t  new_rank = 0;
    vsi_bool ret = FALSE;
    uint32_t dim = inputs[0]->attr.dim_num;
    vsi_size_t multiples[VSI_NN_MAX_DIM_NUM] = { 1, 1, 1, 1 };

    for ( i = 0;  i < dim;  i++)
    {
        multiples[i] = outputs[0]->attr.size[i] / inputs[0]->attr.size[i];
    }

    ret = vsi_nn_kernel_optimize_tile_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            multiples, inputs[0]->attr.dim_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    if (ret)
    {
        if ( _is_supported_axis(shapes[1], new_rank) == FALSE)
        {
            return NULL;
        }

        reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], shapes[2], new_rank );
    }
    else
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[1]->attr.size,
                outputs[0]->attr.dim_num ))
    {
        goto final;
    }

    image_2d = ((reshape_tensors[0]->attr.dim_num == 2 || reshape_tensors[0]->attr.size[2] == 1));
    status = _query_kernel( &reshape_tensors[0], &reshape_tensors[1], image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if( node )
        {
            uint32_t depthIn = (uint32_t)(new_rank > 2 ? reshape_tensors[0]->attr.size[2] : 1);
            uint32_t depthOut = (uint32_t)(new_rank > 2 ? reshape_tensors[1]->attr.size[2] : 1);
            uint32_t batchIn = (uint32_t)(new_rank > 3 ? reshape_tensors[0]->attr.size[3] : 1);

            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    &reshape_tensors[0], 1, &reshape_tensors[1], 1 );

            /* Pass parameters to node. */
            node_params[SCALAR_INPUT_BATCH_IN] = vsi_nn_kernel_scalar_create(
                    graph, I32, &batchIn );
            node_params[SCALAR_INPUT_DEPTH_IN] = vsi_nn_kernel_scalar_create(
                    graph, I32, &depthIn );
            node_params[SCALAR_INPUT_DEPTH_OUT] = vsi_nn_kernel_scalar_create(
                    graph, I32, &depthOut );
            node_params[SCALAR_INPUT_MULTIPLES_0] = vsi_nn_kernel_scalar_create(
                    graph, I32, &multiples[0] );
            node_params[SCALAR_INPUT_MULTIPLES_1] = vsi_nn_kernel_scalar_create(
                    graph, I32, &multiples[1] );
            node_params[SCALAR_INPUT_MULTIPLES_2] = vsi_nn_kernel_scalar_create(
                    graph, I32, &multiples[2] );
            node_params[SCALAR_INPUT_MULTIPLES_3] = vsi_nn_kernel_scalar_create(
                    graph, I32, &multiples[3] );

            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_BATCH_IN] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_DEPTH_IN] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_DEPTH_OUT] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_0] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_1] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_2] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_3] );
        }
    }

final:
    if (reshape_tensors[0] != inputs[0])
    {
        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
    }

    if (reshape_tensors[1] != outputs[0])
    {
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( tile, _setup )
