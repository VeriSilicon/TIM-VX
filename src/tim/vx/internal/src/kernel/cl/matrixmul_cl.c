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
#define KERNEL_SOURCE_1    "matrixmul"
#define KERNEL_SOURCE_2    "matrixmul_transA"
#define KERNEL_SOURCE_3    "matrixmul_cross"

 typedef enum
{
    _2D = 0,
    _3D
} vsi_nn_kernel_image_dim_type_e;

#define HASH_MATRIXMUL_KEY(_type0, _type1, _type2, _image_dim, _trans_a, _cross) \
    ((_type0 << 24) | (_type1 << 16) | (_type2 << 8) | (_image_dim << 4) | (_trans_a << 2) | (_cross))

#define HASH_MATRIXMUL_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_TRANSA_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_transa_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_TRANSB_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_transb_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_SH_KERNEL_MERGE_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_merge")

#define TENSOR_MATRIXMUL_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    { HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 0, 0), \
        HASH_MATRIXMUL_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_TRANSA_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    { HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 1, 0), \
        HASH_MATRIXMUL_TRANSA_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_TRANSB_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    { HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 2, 0), \
        HASH_MATRIXMUL_TRANSB_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_MERGE_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    { HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 0, 2), \
        HASH_MATRIXMUL_SH_KERNEL_MERGE_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } matrixmul_map[] =
{
    TENSOR_MATRIXMUL_KERNELS(F32, F32, F32, _2D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(F32, F32, F32, _3D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(F32, F32, F32, _2D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(F32, F32, F32, _3D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, F32, F32, _2D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, F32, F32, _3D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, I8, F32, _2D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, I8, F32, _3D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(I8, I8, I8, _2D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(I8, I8, I8, _3D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(I8, I8, I8, _2D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(I8, I8, I8, _3D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(I8, I8, I8, _2D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(I8, I8, I8, _3D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, U8, _2D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, U8, _3D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, U8, _2D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, U8, _3D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, U8, _2D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, U8, _3D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, F32, _2D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, F32, _3D,           KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, F32, _2D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, F32, _3D,    KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, F32, _2D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, F32, _3D,    KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_MERGE_KERNELS(U8,  U8, U8,  _3D,    KERNEL_SOURCE_3)
    TENSOR_MATRIXMUL_MERGE_KERNELS(I8,  I8, I8,  _3D,    KERNEL_SOURCE_3)
    TENSOR_MATRIXMUL_MERGE_KERNELS(F32, F32, F32, _3D,   KERNEL_SOURCE_3)
};

/*
 * Kernel params
 */
static vx_param_description_t _matrixmul_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

static vx_param_description_t _matrixmul_merge_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _MATRIXMUL_PARAM_NUM          _cnt_of_array(_matrixmul_kernel_param_def)
#define _MATRIXMUL_MERGE_PARAM_NUM    _cnt_of_array(_matrixmul_merge_kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_matrixmul_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    vsi_size_t width = 0;
    vsi_size_t height = 0;
    vsi_size_t chn = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    width = attr[2]->shape->data[0];
    height = attr[2]->shape->data[1];
    chn = attr[2]->shape->size > 2 ? attr[2]->shape->data[2] : 1;

    if (((attr[0]->shape->size == 4 && attr[1]->shape->size == 3) ||
        (attr[0]->shape->size == 3 && attr[1]->shape->size == 4))
        && attr[0]->shape->data[2] > 1 && attr[1]->shape->data[2] > 1
        && chn == attr[0]->shape->data[2] * attr[1]->shape->data[2])
    {
        if (attr[0]->shape->size == 4)
        {
            chn = attr[1]->shape->data[2];
        }
        else
        {
            chn = attr[0]->shape->data[2];
        }
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = gpu_align_p2((height + gpu_param.global_scale[1] - 1)
                                        / gpu_param.global_scale[1], 4);
    gpu_param.global_size[2]   = chn;

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
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _matrixmul_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_size_t depth,
    int32_t transa,
    int32_t cross
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input1_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_nn_kernel_image_dim_type_e dim_type = _2D;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    if (depth > 1)
    {
        dim_type = _3D;
    }

    if (input0_dtype == I16 || input0_dtype == I32)
    {
        input0_dtype = I8;
    }
    else if (input0_dtype == F16)
    {
        input0_dtype = F32;
    }
    else if (input0_dtype == U32)
    {
        input0_dtype = U8;
    }

    if (input1_dtype == I16 || input1_dtype == I32)
    {
        input1_dtype = I8;
    }
    else if (input1_dtype == F16)
    {
        input1_dtype = F32;
    }
    else if (input1_dtype == U32)
    {
        input1_dtype = U8;
    }

    if (output_dtype == I16 || output_dtype == I32)
    {
        output_dtype = I8;
    }
    else if (output_dtype == F16)
    {
        output_dtype = F32;
    }
    else if (output_dtype == U32)
    {
        output_dtype = U8;
    }

    key = HASH_MATRIXMUL_KEY( input0_dtype, input1_dtype, output_dtype, dim_type, transa, cross );

    for( i = 0; i < _cnt_of_array(matrixmul_map); i ++ )
    {
        if ( matrixmul_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(matrixmul_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  matrixmul_map[i].function_name );
        if (cross == 0)
        {
            kernel->info.parameters = _matrixmul_kernel_param_def;
            kernel->info.numParams = _cnt_of_array( _matrixmul_kernel_param_def );
        }
        else if (cross == 2)
        {
            kernel->info.parameters = _matrixmul_merge_kernel_param_def;
            kernel->info.numParams = _cnt_of_array( _matrixmul_merge_kernel_param_def );
        }
        kernel->info.initialize = _matrixmul_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                matrixmul_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                matrixmul_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_MATRIXMUL_MERGE_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t transposeA  = vsi_nn_kernel_param_get_int32( params, "transposeA" );
    int32_t transposeB  = vsi_nn_kernel_param_get_int32( params, "transposeB" );
    int32_t cross_flg  = vsi_nn_kernel_param_get_int32( params, "cross_flg" );
    int32_t transFlg    = 0;
    vsi_size_t M = inputs[0]->attr.size[1];
    vsi_size_t K = inputs[0]->attr.size[0];
    vsi_size_t N = inputs[1]->attr.size[0];
    vsi_size_t a_depth = 0;
    vsi_size_t b_depth = 0;
    vsi_size_t depth = outputs[0]->attr.dim_num > 2 ? outputs[0]->attr.size[2] : 1;
    uint32_t ac2zero = 0;
    uint32_t bc2zero = 0;
    float    scale_a = vsi_nn_get_tensor_scale(inputs[0]);
    float    zp_a = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float    scale_b = vsi_nn_get_tensor_scale(inputs[1]);
    float    zp_b = (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    float    scale_out = vsi_nn_get_tensor_scale(outputs[0]);
    float    zp_out = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    int32_t  outer = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    scale_out = 1 / scale_out;

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if (transposeB)
    {
        N = inputs[1]->attr.size[1];
        transFlg = 2;
    }

    if (transposeA)
    {
        K = inputs[0]->attr.size[1];
        M = inputs[0]->attr.size[0];
        transFlg = 1;
    }

    a_depth = inputs[0]->attr.dim_num > 2 ? inputs[0]->attr.size[2] : 1;
    b_depth = inputs[1]->attr.dim_num > 2 ? inputs[1]->attr.size[2] : 1;

    if (b_depth == 1)
    {
        bc2zero = 1;
    }
    if (a_depth == 1)
    {
        ac2zero = 1;
    }

    if (inputs[0]->attr.dim_num == 4 && inputs[1]->attr.dim_num == 3
        && a_depth > 1 && b_depth > 1 && cross_flg == 2)
    {
        ac2zero = 1;
        bc2zero = 0;
        outer = (int32_t)a_depth;
    }
    else if (inputs[1]->attr.dim_num == 4 && inputs[0]->attr.dim_num == 3
        && a_depth > 1 && b_depth > 1 && cross_flg == 2)
    {
        ac2zero = 0;
        bc2zero = 1;
        outer = (int32_t)b_depth;
    }

    status = _query_kernel( kernel, inputs, outputs, depth, transFlg, cross_flg );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 3;
            size_t param_num = cross_flg == 2 ? _MATRIXMUL_MERGE_PARAM_NUM : _MATRIXMUL_PARAM_NUM;
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, param_num,
                    inputs, 2, outputs, 1 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &M );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &K );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &N );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ac2zero );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &bc2zero );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_a );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &zp_a );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_b );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &zp_b );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_out );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &zp_out );
            if (cross_flg == 2)
            {
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &outer );
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, param_num );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
            vsi_nn_kernel_scalar_release( &node_params[13] );
            if (cross_flg == 2)
            {
                vsi_nn_kernel_scalar_release( &node_params[14] );
            }
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( matrixmul, _setup )
