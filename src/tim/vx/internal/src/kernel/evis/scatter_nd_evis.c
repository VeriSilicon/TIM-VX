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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "scatter_nd"
#define KERNEL_SOURCE_2    "scatter_nd_big"

#define HASH_SCATTER_ND_KEY(_input0_type, _output_type, _coord_dim, _reshape_type) \
    ((_input0_type << 24) | (_output_type << 16) | (_coord_dim << 8) | (_reshape_type))

#define HASH_SCATTER_ND_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_SCATTER_ND_SH_KERNEL_BIG_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_"#SRC0_TYPE"to"#DST_TYPE"_big")

#define TENSOR_SCATTER_ND_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_KEY(IN1_TYPE, OUT_TYPE, 0, 0), \
        HASH_SCATTER_ND_SH_KERNEL_NAME(IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_BIG_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_KEY(IN1_TYPE, OUT_TYPE, 0, 1), \
        HASH_SCATTER_ND_SH_KERNEL_BIG_NAME(IN1_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } scatter_nd_map[] =
{
    TENSOR_SCATTER_ND_KERNELS(I32, I8,  I8,       KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, U8,  U8,       KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, I16, I16,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_KERNELS(I32, F16, F16,      KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_BIG_KERNELS(I32, I8,  I8,   KERNEL_SOURCE_2)
    TENSOR_SCATTER_ND_BIG_KERNELS(I32, U8,  U8,   KERNEL_SOURCE_2)
    TENSOR_SCATTER_ND_BIG_KERNELS(I32, I16, I16,  KERNEL_SOURCE_2)
    TENSOR_SCATTER_ND_BIG_KERNELS(I32, F16, F16,  KERNEL_SOURCE_2)
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
    // Add kererl parameters here
};
#define _SCATTER_ND_PARAM_NUM  _cnt_of_array( _scatter_nd_kernel_param_def )

static vsi_status get_scatter_nd_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t coordDim,
    vsi_size_t* width,
    vsi_size_t* area,
    int32_t* newDim,
    int32_t* isBig
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

#define VSI_NN_MAX_IMAGE_WIDTH  (65536)

    newDim[0] = 0;
    for(i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for(i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    sizes[0] = block_size;
    sizes[1] = elementCnt / block_size;
    newDim[0] = 2;

    if((elementCnt / block_size) >= VSI_NN_MAX_IMAGE_WIDTH)
    {
        isBig[0] |= 1;
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

    return VSI_SUCCESS;
} /* _get_EltOP_tensor_reshape_size */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_scatter_nd_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t     block_size = 1;
    int32_t     height     = 1;
    int32_t     index_num  = 1;
    int32_t     output_zp  = 0;
    int32_t     width = 0, area = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size = (int32_t)(attr[2]->shape->data[0]);
    height     = (int32_t)(attr[2]->shape->data[1]);
    index_num  = (int32_t)(attr[0]->shape->data[1]);
    output_zp  = (int32_t)(attr[2]->asymm.zero_point);

    if(coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
    }
    else if(coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
    }
    else if(coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
    }

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((block_size + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniAccumulateSum_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param( node,
            "uniAccumulateSum_2x8", &uniAccumulateSum_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "index_num", &index_num );
        status |= vsi_nn_kernel_gpu_add_param( node, "zeropoint", &output_zp );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
    }

OnError:
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
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _scatter_nd_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_big_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t     block_size = 1;
    int32_t     height     = 1;
    int32_t     index_num  = 1;
    int32_t     output_zp  = 0;
    int32_t     width = 0, area = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size = (int32_t)(attr[2]->shape->data[0]);
    height     = (int32_t)(attr[2]->shape->data[1]);
    index_num  = (int32_t)(attr[0]->shape->data[1]);
    output_zp  = (int32_t)(attr[2]->asymm.zero_point);

    if(coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
    }
    else if(coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
    }
    else if(coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniAccumulateSum_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param( node,
            "uniAccumulateSum_2x8", &uniAccumulateSum_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "index_num", &index_num );
        status |= vsi_nn_kernel_gpu_add_param( node, "update_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "zeropoint", &output_zp );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
    }

OnError:
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
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _scatter_nd_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t coord_dim,
    int32_t isBig
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input1_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_SCATTER_ND_KEY( input1_dtype, output_dtype, 0, isBig );

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
        if(isBig)
        {
            kernel->info.initialize = _scatter_nd_big_initializer;
        }
        else
        {
            kernel->info.initialize = _scatter_nd_initializer;
        }

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_SCATTER_ND_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t coord_dim   = vsi_nn_kernel_param_get_int32( params, "coord_dim" );
    int32_t rs_in_dim = 0, rs_idx_dim = 0, rs_out_dim = 0;
    vsi_size_t width = 0, area = 0;
    int32_t big_flg = 0;

    status = get_scatter_nd_tensor_reshape_size(&inputs[0], shapes[0], coord_dim, 0,
                                                    NULL, NULL, &rs_idx_dim, &big_flg);
    status |= get_scatter_nd_tensor_reshape_size(&inputs[1], shapes[1], block_size, 0,
                                                    NULL, NULL, &rs_in_dim, &big_flg);
    status |= get_scatter_nd_tensor_reshape_size(&outputs[0], shapes[2], block_size, coord_dim,
                                                    &width, &area, &rs_out_dim, &big_flg);
    if(status != VSI_SUCCESS)
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, coord_dim, big_flg);
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 0;
            /* Pass parameters to node. */
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[0], rs_in_dim );
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[1], rs_idx_dim );
            //tmp_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &area );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _SCATTER_ND_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_tensor_release( &tmp_params[0] );
            vsi_nn_kernel_tensor_release( &tmp_params[1] );
            vsi_nn_kernel_tensor_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_scalar_release( &tmp_params[5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( scatter_nd, _setup )

