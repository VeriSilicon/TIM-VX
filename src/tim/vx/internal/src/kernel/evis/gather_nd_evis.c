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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "gather_nd"
#define KERNEL_SOURCE_2    "gather_nd_2d"
#define KERNEL_SOURCE_3    "gather_nd_3d"
#define KERNEL_SOURCE_4    "gather_nd_mix"
#define KERNEL_SOURCE_5    "gather_nd_2d_mix"
#define KERNEL_SOURCE_6    "gather_nd_3d_mix"
#define KERNEL_SOURCE_7    "gather_nd_batch"
#define KERNEL_SOURCE_8    "gather_nd_batch_2d"

 typedef enum
{
    _error = -1,
    _1D = 0,
    _2D,
    _3D
} vsi_nn_kernel_coord_type_e;

#define HASH_GATHER_ND_KEY(_input0_type, _output_type, _coord_dim, _batch_dim) \
    ((_input0_type << 24) | (_output_type << 16) | (_coord_dim << 8) | (_batch_dim))

#define HASH_GATHER_ND_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE, COORD_TYPE) \
    CVIVANTE_NAMESPACE("evis.gather_nd_"#SRC0_TYPE"to"#DST_TYPE#COORD_TYPE)

#define TENSOR_GATHER_ND_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, COORD_TYPE, SOURCE) \
    { HASH_GATHER_ND_KEY(IN0_TYPE, OUT_TYPE, COORD_TYPE, 0), \
        HASH_GATHER_ND_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE, COORD_TYPE), \
        SOURCE },

#define HASH_GATHER_ND_BATCH_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE, COORD_TYPE) \
    CVIVANTE_NAMESPACE("evis.gather_nd_batch_"#SRC0_TYPE"to"#DST_TYPE#COORD_TYPE)

#define TENSOR_GATHER_ND_BATCH_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, COORD_TYPE, SOURCE) \
    { HASH_GATHER_ND_KEY(IN0_TYPE, OUT_TYPE, COORD_TYPE, 1), \
        HASH_GATHER_ND_BATCH_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE, COORD_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } gather_nd_map[] =
{
    TENSOR_GATHER_ND_KERNELS(I8,  I32, I8,  _1D,      KERNEL_SOURCE_1)
    TENSOR_GATHER_ND_KERNELS(U8,  I32, U8,  _1D,      KERNEL_SOURCE_1)
    TENSOR_GATHER_ND_KERNELS(I16, I32, I16, _1D,      KERNEL_SOURCE_1)
    TENSOR_GATHER_ND_KERNELS(F16, I32, F16, _1D,      KERNEL_SOURCE_1)
    TENSOR_GATHER_ND_KERNELS(I8,  I32, I8,  _2D,      KERNEL_SOURCE_2)
    TENSOR_GATHER_ND_KERNELS(U8,  I32, U8,  _2D,      KERNEL_SOURCE_2)
    TENSOR_GATHER_ND_KERNELS(I16, I32, I16, _2D,      KERNEL_SOURCE_2)
    TENSOR_GATHER_ND_KERNELS(F16, I32, F16, _2D,      KERNEL_SOURCE_2)
    TENSOR_GATHER_ND_KERNELS(I8,  I32, I8,  _3D,      KERNEL_SOURCE_3)
    TENSOR_GATHER_ND_KERNELS(U8,  I32, U8,  _3D,      KERNEL_SOURCE_3)
    TENSOR_GATHER_ND_KERNELS(I16, I32, I16, _3D,      KERNEL_SOURCE_3)
    TENSOR_GATHER_ND_KERNELS(F16, I32, F16, _3D,      KERNEL_SOURCE_3)

    TENSOR_GATHER_ND_KERNELS(I8,  I32, F16, _1D,      KERNEL_SOURCE_4)
    TENSOR_GATHER_ND_KERNELS(I16, I32, F16, _1D,      KERNEL_SOURCE_4)
    TENSOR_GATHER_ND_KERNELS(F16, I32, I8,  _1D,      KERNEL_SOURCE_4)
    TENSOR_GATHER_ND_KERNELS(F16, I32, I16, _1D,      KERNEL_SOURCE_4)
    TENSOR_GATHER_ND_KERNELS(U8,  I32, F16, _1D,      KERNEL_SOURCE_4)
    TENSOR_GATHER_ND_KERNELS(F16, I32, U8,  _1D,      KERNEL_SOURCE_4)

    TENSOR_GATHER_ND_KERNELS(I8,  I32, F16, _2D,      KERNEL_SOURCE_5)
    TENSOR_GATHER_ND_KERNELS(I16, I32, F16, _2D,      KERNEL_SOURCE_5)
    TENSOR_GATHER_ND_KERNELS(F16, I32, I8,  _2D,      KERNEL_SOURCE_5)
    TENSOR_GATHER_ND_KERNELS(F16, I32, I16, _2D,      KERNEL_SOURCE_5)
    TENSOR_GATHER_ND_KERNELS(U8,  I32, F16, _2D,      KERNEL_SOURCE_5)
    TENSOR_GATHER_ND_KERNELS(F16, I32, U8,  _2D,      KERNEL_SOURCE_5)

    TENSOR_GATHER_ND_KERNELS(I8,  I32, F16, _3D,      KERNEL_SOURCE_6)
    TENSOR_GATHER_ND_KERNELS(I16, I32, F16, _3D,      KERNEL_SOURCE_6)
    TENSOR_GATHER_ND_KERNELS(F16, I32, I8,  _3D,      KERNEL_SOURCE_6)
    TENSOR_GATHER_ND_KERNELS(F16, I32, I16, _3D,      KERNEL_SOURCE_6)
    TENSOR_GATHER_ND_KERNELS(U8,  I32, F16, _3D,      KERNEL_SOURCE_6)
    TENSOR_GATHER_ND_KERNELS(F16, I32, U8,  _3D,      KERNEL_SOURCE_6)

    TENSOR_GATHER_ND_BATCH_KERNELS(I8,  I32, I8,  _1D,      KERNEL_SOURCE_7)
    TENSOR_GATHER_ND_BATCH_KERNELS(U8,  I32, U8,  _1D,      KERNEL_SOURCE_7)
    TENSOR_GATHER_ND_BATCH_KERNELS(I16, I32, I16, _1D,      KERNEL_SOURCE_7)
    TENSOR_GATHER_ND_BATCH_KERNELS(F16, I32, F16, _1D,      KERNEL_SOURCE_7)
    TENSOR_GATHER_ND_BATCH_KERNELS(I8,  I32, I8,  _2D,      KERNEL_SOURCE_8)
    TENSOR_GATHER_ND_BATCH_KERNELS(U8,  I32, U8,  _2D,      KERNEL_SOURCE_8)
    TENSOR_GATHER_ND_BATCH_KERNELS(I16, I32, I16, _2D,      KERNEL_SOURCE_8)
    TENSOR_GATHER_ND_BATCH_KERNELS(F16, I32, F16, _2D,      KERNEL_SOURCE_8)
};

/*
 * Kernel params
 */
static vx_param_description_t _gather_nd_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GATHER_ND_PARAM_NUM  _cnt_of_array( _gather_nd_kernel_param_def )

static vsi_status get_gather_nd_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    vsi_size_t block_size,
    uint32_t coordDim,
    int32_t* newDim,
    uint32_t  batch_dims
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    vsi_size_t elementCnt = 1;
#define VSI_NN_MAX_IMAGE_WIDTH  GPU_TENSOR_MAX_WIDTH

    newDim[0] = 0;
    for (i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    if (coordDim) // input reshape
    {
        uint32_t offset = dims_num - coordDim + 1 - batch_dims;

        if (batch_dims)
        {
            int32_t rank = 1;
            for (i = 0; i < offset; i++)
            {
                sizes[0] *= input_size[i];
            }

            for (i = 0; i < coordDim - 1; i++)
            {
                sizes[rank++] = input_size[i + offset];
            }

            for (i = 0; i < batch_dims; i++)
            {
                sizes[rank] *= input_size[dims_num - i - 1];
            }

            newDim[0] = rank + 1;
        }
        else
        {
            for (i = coordDim-1; i > 0; i--)
            {
                sizes[i] = input_size[i + offset - 1];
            }
            for (i = 0; i < offset; i++)
            {
                sizes[0] *= input_size[i];
            }

            newDim[0] = coordDim;
            if (coordDim == 1)
            {
                newDim[0] = 2;
                sizes[0] = block_size;
                sizes[1] = elementCnt / block_size;
            }
            else if (coordDim == 4)
            {
                newDim[0] = 3;
            }
        }

        status = VSI_SUCCESS;
    }
    else  // indices&output reshape
    {
        if ((elementCnt / block_size) < VSI_NN_MAX_IMAGE_WIDTH && batch_dims == 0)
        {
            sizes[0] = block_size;
            sizes[1] = elementCnt / block_size;
            status = VSI_SUCCESS;
            newDim[0] = 2;
        }
        else if (batch_dims > 0)
        {
            vsi_size_t batch_cnt = 1;
            for (i = 0; i < batch_dims; ++i)
            {
                batch_cnt *= input_size[dims_num - i - 1];
            }

            sizes[0] = block_size;
            sizes[1] = (elementCnt / block_size) / batch_cnt;
            sizes[2] = batch_cnt;
            status = VSI_SUCCESS;
            newDim[0] = 3;
        }
    }
#undef VSI_NN_MAX_IMAGE_WIDTH

    return status;
} /* _get_EltOP_tensor_reshape_size */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_gather_nd_initializer)
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
    int32_t     block_size  = 0;
    int32_t     indices_num = 1;
    int32_t     batch_num   = 1;
    int32_t     src0ZP      = 0;
    float       src0Scale   = 1;
    int32_t     dstZP       = 0;
    float       dstScale    = 1;

    uint32_t pack_key = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &block_size);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    src0Scale = attr[0]->scale;
    src0ZP    = attr[0]->zero_point;
    dstScale  = 1.0f / attr[2]->scale;
    dstZP     = attr[2]->zero_point;

    indices_num = (int32_t)(attr[1]->shape->data[1]);
    batch_num = (int32_t)(attr[1]->shape->size > 2 ? attr[1]->shape->data[2] : 1);

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((block_size + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = indices_num;
    gpu_param.global_size[2]   = batch_num;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE)    \
        (IN0_TYPE | (OUT_TYPE << 8))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[2]->dtype );
    {
        uint16_t M0               = 0;
        int32_t  postShift        = 0;
        uint32_t multAndoutZP0[2] = {0};
        uint32_t multAndoutZP1[2] = {0};
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertFp16toU8_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8, F16 ):
        case _PACK_SELECT_KEY( I8, F16 ):
        case _PACK_SELECT_KEY( I16, F16 ):
            {
                gpu_quantize_multiplier_16bit( (double)src0Scale * dstScale, &M0, &postShift);
                multAndoutZP0[0] = (uint32_t)(M0);
                multAndoutZP0[1] = (uint32_t)((dstZP << postShift) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift );
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8 ):
        case _PACK_SELECT_KEY( F16, I8 ):
        case _PACK_SELECT_KEY( F16, I16 ):
            {
                int32_t  postShift0       = 0;
                gpu_quantize_multiplier_16bit( (double)src0Scale * dstScale, &M0, &postShift0);

                multAndoutZP1[0] = (uint32_t)(M0);
                multAndoutZP1[1] = (uint32_t)((dstZP << postShift0) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniConvertFp16toU8_2x8, postShift0 );
                status = vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertFp16toU8_2x8", &uniConvertFp16toU8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }
#undef _PACK_SELECT_KEY

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
} /* _gather_nd_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t coord_dim,
    int32_t batch_dims
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_nn_kernel_coord_type_e coord_type = _error;
    uint32_t key = 0;
    int32_t batch_flg = batch_dims > 0 ? 1 : 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    if (input0_dtype == BF16)
    {
        input0_dtype = F16;
    }
    if (output_dtype == BF16)
    {
        output_dtype = F16;
    }

    if (coord_dim == 1)
    {
        coord_type = _1D;
    }
    else if (coord_dim == 2)
    {
        coord_type = _2D;
    }
    else if (coord_dim == 3 || coord_dim == 4)
    {
        coord_type = _3D;
    }

    key = HASH_GATHER_ND_KEY( input0_dtype, output_dtype, coord_type, batch_flg );

    for ( i = 0; i < _cnt_of_array(gather_nd_map); i ++ )
    {
        if ( gather_nd_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(gather_nd_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  gather_nd_map[i].function_name );
        kernel->info.parameters = _gather_nd_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _gather_nd_kernel_param_def );
        kernel->info.initialize = _gather_nd_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                gather_nd_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                gather_nd_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_GATHER_ND_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t batch_dims  = vsi_nn_kernel_param_get_int32( params, "batch_dims" );
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t coord_dim   = vsi_nn_kernel_param_get_int32( params, "coord_dim" );
    int32_t rs_in_dim = 0, rs_idx_dim = 0, rs_out_dim = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = get_gather_nd_tensor_reshape_size(&inputs[0], shapes[0], block_size, coord_dim, &rs_in_dim, batch_dims);
    status |= get_gather_nd_tensor_reshape_size(&inputs[1], shapes[1], coord_dim, 0, &rs_idx_dim, batch_dims);
    status |= get_gather_nd_tensor_reshape_size(&outputs[0], shapes[2], block_size, 0, &rs_out_dim, batch_dims);
    if (status != VSI_SUCCESS)
    {
        return NULL;
    }

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, coord_dim, batch_dims );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 0;
            /* Pass parameters to node. */
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[0], rs_in_dim );
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[1], rs_idx_dim );
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _GATHER_ND_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_tensor_release( &tmp_params[0] );
            vsi_nn_kernel_tensor_release( &tmp_params[1] );
            vsi_nn_kernel_tensor_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( gather_nd, _setup )
