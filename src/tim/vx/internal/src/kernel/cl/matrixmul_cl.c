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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "matrixmul"
#define KERNEL_SOURCE_2    "matrixmul_transA"
#define KERNEL_SOURCE_3    "matrixmul_cross"
#define KERNEL_SOURCE_4    "matrixmul_4x"

 typedef enum
{
    _2D = 0,
    _3D
} vsi_nn_kernel_image_dim_type_e;

#define HASH_MATRIXMUL_KEY(_type0, _type1, _type2, _image_dim, flag_4x, _trans_a, _cross) \
    ((_type0 << 24) | (_type1 << 16) | (_type2 << 8) | (_image_dim << 6) | \
     (flag_4x << 4) | (_trans_a << 2) | (_cross))

#define HASH_MATRIXMUL_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_TRANSA_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_transa_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_TRANSB_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_transb_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_SH_KERNEL_MERGE_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_merge")

#define HASH_MATRIXMUL_4X_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_4x_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_4X_TRANSA_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_4x_transa_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define HASH_MATRIXMUL_4X_TRANSA_LOCAL_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, IMAGE_DIM) \
    CVIVANTE_NAMESPACE("cl.gemm_4x_transa_local_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE#IMAGE_DIM)

#define TENSOR_MATRIXMUL_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    { HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 0, 0, 0), \
        HASH_MATRIXMUL_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_4X_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    {HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 1, 0, 0), \
        HASH_MATRIXMUL_4X_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_4X_TRANSA_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    {HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 1, 1, 0), \
        HASH_MATRIXMUL_4X_TRANSA_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_4X_TRANSA_LOCAL_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    {HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 2, 1, 0), \
        HASH_MATRIXMUL_4X_TRANSA_LOCAL_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_TRANSA_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    {HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 0, 1, 0), \
        HASH_MATRIXMUL_TRANSA_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_TRANSB_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    {HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 0, 2, 0), \
        HASH_MATRIXMUL_TRANSB_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM), \
        SOURCE },

#define TENSOR_MATRIXMUL_MERGE_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, SOURCE) \
    {HASH_MATRIXMUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, IMAGE_DIM, 0, 0, 2), \
        HASH_MATRIXMUL_SH_KERNEL_MERGE_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } matrixmul_map[] =
{
    TENSOR_MATRIXMUL_KERNELS(F32, F32, F32, _2D,            KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(F32, F32, F32, _3D,            KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(F32, F32, F32, _2D,     KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(F32, F32, F32, _3D,     KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, F32, F32, _2D,     KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, F32, F32, _3D,     KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, I8, F32, _2D,      KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(F32, I8, F32, _3D,      KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(I8, I8, I8, _2D,               KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(I8, I8, I8, _3D,               KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(I8, I8, I8, _2D,        KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(I8, I8, I8, _3D,        KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(I8, I8, I8, _2D,        KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(I8, I8, I8, _3D,        KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, U8, _2D,               KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, U8, _3D,               KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, U8, _2D,        KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, U8, _3D,        KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, U8, _2D,        KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, U8, _3D,        KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, F32, _2D,              KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_KERNELS(U8, U8, F32, _3D,              KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, F32, _2D,       KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSA_KERNELS(U8, U8, F32, _3D,       KERNEL_SOURCE_2)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, F32, _2D,       KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_TRANSB_KERNELS(U8, U8, F32, _3D,       KERNEL_SOURCE_1)
    TENSOR_MATRIXMUL_MERGE_KERNELS(U8,  U8, U8,  _3D,       KERNEL_SOURCE_3)
    TENSOR_MATRIXMUL_MERGE_KERNELS(I8,  I8, I8,  _3D,       KERNEL_SOURCE_3)
    TENSOR_MATRIXMUL_MERGE_KERNELS(F32, F32, F32, _3D,      KERNEL_SOURCE_3)
    TENSOR_MATRIXMUL_4X_KERNELS(F32, F32, F32, _2D,         KERNEL_SOURCE_4)
    TENSOR_MATRIXMUL_4X_TRANSA_KERNELS(F32, F32, F32, _2D,  KERNEL_SOURCE_4)
    TENSOR_MATRIXMUL_4X_TRANSA_LOCAL_KERNELS(F32, F32, F32, _2D,  KERNEL_SOURCE_4)
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

DEF_KERNEL_INITIALIZER(_matrixmul_4x_initializer)
(vsi_nn_kernel_node_t node,
 const vsi_nn_kernel_node_param_t* param,
 size_t param_size) {
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {3, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    vsi_nn_kernel_tensor_attr_t* attr = NULL;
    vsi_size_t width = 0;
    vsi_size_t height = 0;

    VSI_UNREFERENCED(param_size);

    attr =
        vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[2]);
    CHECK_PTR_FAIL_GOTO(attr, "Create tensor attr buffer fail.", final);

    width  = attr->shape->data[0];
    height = attr->shape->data[1];

    gpu_param.dim = 2;
    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;

    gpu_param.global_size[0] = (width + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0];
    gpu_param.global_size[1] = height;
    gpu_param.global_size[2] = 1;

    status = vsi_nn_kernel_gpu_config(node, &gpu_param);
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr) {
        vsi_nn_kernel_tensor_attr_release(&attr);
        attr = NULL;
    }
    return status;
} /* _matrixmul_4x_initializer() */

DEF_KERNEL_INITIALIZER(_matrixmul_4x_local_initializer)
(vsi_nn_kernel_node_t node,
 const vsi_nn_kernel_node_param_t* param,
 size_t param_size) {
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {3, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    vsi_nn_kernel_tensor_attr_t* attr = NULL;
    vsi_size_t width = 0;


    VSI_UNREFERENCED(param_size);

    attr = vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[2]);
    CHECK_PTR_FAIL_GOTO(attr, "Create tensor attr buffer fail.", final);

    width = attr->shape->data[0];

    gpu_param.dim = 2;
    gpu_param.local_size[0] = 1;
    gpu_param.local_size[1] = 64;
    gpu_param.local_size[2] = 1;

    gpu_param.global_scale[0] = 16;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;

    gpu_param.global_size[0] =
        (width + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0];
    gpu_param.global_size[1] = 64;
    gpu_param.global_size[2] = 1;

    status = vsi_nn_kernel_gpu_config(node, &gpu_param);
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr) {
        vsi_nn_kernel_tensor_attr_release(&attr);
        attr = NULL;
    }
    return status;
} /* _matrixmul_4x_local_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_size_t depth,
    int32_t flag_4x,
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

    key = HASH_MATRIXMUL_KEY( input0_dtype, input1_dtype, output_dtype, dim_type, flag_4x, transa, cross );

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

        if ((flag_4x == 2) && (transa == 1)) {
            kernel->info.initialize = _matrixmul_4x_local_initializer;
        }
        else if (flag_4x == 1) {
            kernel->info.initialize = _matrixmul_4x_initializer;
        } else {
            kernel->info.initialize = _matrixmul_initializer;
        }

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
    int32_t transFlg    = 0;
    int32_t flag_4x = 0;
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
    vsi_size_t final_shape[VSI_NN_MAX_DIM_NUM] = {1, 1, 1, 1};
    uint32_t final_rank = 0;
    vsi_nn_tensor_t* rs_in_tensors = NULL;
    vsi_nn_tensor_t* rs_out_tensors = NULL;
    vsi_nn_tensor_t* final_in_tensors[2] = {NULL};
    vsi_nn_tensor_t* final_out_tensors[1] = {NULL};
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input1_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    uint32_t new_rank[3] = {0};
    uint32_t cross_flg = 0;
    uint32_t size_axis_in_out[3] = {0};
    uint32_t stride_axis_in_out[9] = {0};
    vsi_nn_tensor_t* tmp_inputs[2]  = {NULL};
    vsi_nn_tensor_t* tmp_outputs[1] = {NULL};
    vsi_bool shader_cnt_support = FALSE;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    scale_out = 1 / scale_out;

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = vsi_nn_kernel_optimize_matrixmul_broadcast_shape(
                                       inputs[0]->attr.size,
                                       inputs[1]->attr.size,
                                       outputs[0]->attr.size,
                                       inputs[0]->attr.dim_num,
                                       inputs[1]->attr.dim_num,
                                       outputs[0]->attr.dim_num,
                                       shapes[0], shapes[1], shapes[2], new_rank,
                                       &cross_flg, size_axis_in_out, stride_axis_in_out);
    if (status)
    {
        tmp_inputs[0] = vsi_nn_reshape_tensor(graph, inputs[0], shapes[0], new_rank[0]);
        tmp_inputs[1] = vsi_nn_reshape_tensor(graph, inputs[1], shapes[1], new_rank[1]);
        tmp_outputs[0] = vsi_nn_reshape_tensor(graph, outputs[0], shapes[2], new_rank[2]);

        M = tmp_inputs[0]->attr.size[1];
        K = tmp_inputs[0]->attr.size[0];
        N = tmp_inputs[1]->attr.size[0];
        depth = tmp_outputs[0]->attr.dim_num > 2 ? tmp_outputs[0]->attr.size[2] : 1;
    }
    else
    {
        VSILOGE("illegal inputs shape");
        status = VSI_FAILURE;
        goto final;
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

    a_depth = tmp_inputs[0]->attr.dim_num > 2 ? tmp_inputs[0]->attr.size[2] : 1;
    b_depth = tmp_inputs[1]->attr.dim_num > 2 ? tmp_inputs[1]->attr.size[2] : 1;

    if (b_depth == 1)
    {
        bc2zero = 1;
    }
    if (a_depth == 1)
    {
        ac2zero = 1;
    }

    if (tmp_inputs[0]->attr.dim_num == 4 && tmp_inputs[1]->attr.dim_num == 3
        && a_depth > 1 && b_depth > 1 && cross_flg == 2)
    {
        ac2zero = 1;
        bc2zero = 0;
        outer = (int32_t)a_depth;
    }
    else if (tmp_inputs[1]->attr.dim_num == 4 && tmp_inputs[0]->attr.dim_num == 3
        && a_depth > 1 && b_depth > 1 && cross_flg == 2)
    {
        ac2zero = 0;
        bc2zero = 1;
        outer = (int32_t)b_depth;
    }

    final_in_tensors[0]  = tmp_inputs[0];
    final_in_tensors[1]  = tmp_inputs[1];
    final_out_tensors[0] = tmp_outputs[0];

    input0_dtype = vsi_nn_kernel_map_dtype(tmp_inputs[0]->attr.dtype.vx_type);
    input1_dtype = vsi_nn_kernel_map_dtype(tmp_inputs[1]->attr.dtype.vx_type);
    output_dtype = vsi_nn_kernel_map_dtype(tmp_outputs[0]->attr.dtype.vx_type);


    if (((transFlg == 0) || (transFlg == 1)) && (cross_flg == 0) &&
        (F32 == input0_dtype) && (F32 == input1_dtype) && (F32 == output_dtype))
    {
        vsi_size_t in1_w = tmp_inputs[1]->attr.size[0];
        vsi_size_t in1_h = tmp_inputs[1]->attr.size[1];
        vsi_size_t in1_c = tmp_inputs[1]->attr.dim_num > 2 ? tmp_inputs[1]->attr.size[2] : 1;
        vsi_size_t in1_n = tmp_inputs[1]->attr.dim_num > 3 ? tmp_inputs[1]->attr.size[3] : 1;
        vsi_size_t out_w = tmp_outputs[0]->attr.size[0];
        vsi_size_t out_h = tmp_outputs[0]->attr.size[1];
        vsi_size_t out_c = tmp_outputs[0]->attr.dim_num > 2 ? tmp_outputs[0]->attr.size[2] : 1;
        vsi_size_t out_n = tmp_outputs[0]->attr.dim_num > 3 ? tmp_outputs[0]->attr.size[3] : 1;
        if ((in1_w == 1) && (in1_h % 4 == 0) && (in1_c == 1) && (in1_n == 1) &&
            (out_w == 1) && (out_h % 4 == 0) && (out_c == 1) && (out_n == 1))
        {
            final_shape[0] = in1_h;
            final_shape[1] = in1_w;
            final_rank = 2;
            rs_in_tensors = vsi_nn_reshape_tensor(graph, tmp_inputs[1], final_shape, final_rank);
            final_in_tensors[1] = rs_in_tensors;

            final_shape[0] = out_h;
            final_shape[1] = out_w;
            final_rank = 2;
            rs_out_tensors = vsi_nn_reshape_tensor(graph, tmp_outputs[0], final_shape, final_rank);
            final_out_tensors[0] = rs_out_tensors;


#if VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT
            shader_cnt_support =
                (graph->ctx->config.subGroupSize >= 64 && graph->ctx->config.use_40bits_va) ? TRUE : FALSE;
#endif
            if ((in1_h % 64 == 0) && (transFlg == 1) && (out_h % 8 == 0) && shader_cnt_support)
            {
                flag_4x = 2;
            }
            else
            {
                flag_4x = 1;
            }

        }
    }

    status = _query_kernel(kernel, tmp_inputs, tmp_outputs, depth, flag_4x, transFlg, cross_flg);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 3;
            size_t param_num = cross_flg == 2 ? _MATRIXMUL_MERGE_PARAM_NUM : _MATRIXMUL_PARAM_NUM;
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, param_num,
                    final_in_tensors, 2, final_out_tensors, 1 );
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

final:
    vsi_safe_release_tensor(tmp_inputs[0]);
    vsi_safe_release_tensor(tmp_inputs[1]);
    vsi_safe_release_tensor(tmp_outputs[0]);
    vsi_safe_release_tensor(rs_in_tensors);
    vsi_safe_release_tensor(rs_out_tensors);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( matrixmul, _setup )
