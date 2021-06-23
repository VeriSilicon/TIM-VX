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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_SUM_SQR,
    INTERNAL_KERNEL_MEAN_VARI,
    INTERNAL_KERNEL_NORM,
} _internal_kernel_e;

#define KERNEL_SOURCE_1    "group_normalization_u8"
#define KERNEL_SOURCE_2    "group_normalization_f32"
#define KERNEL_SOURCE_3    "group_normalization_i32"

// Add kernel hashtable here
#define HASH_GROUPNORM_SUM_SQR_KERNEL_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("cl.group_norm_sumsqr_"#SRC0_TYPE)

#define HASH_GROUPNORM_SUM_SQR_KERNEL_2D_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("cl.group_norm_sumsqr_"#SRC0_TYPE"_2D")

#define HASH_GROUPNORM_MEAN_VARI_KERNEL_NAME \
    CVIVANTE_NAMESPACE("cl.group_norm_meanvari")

#define HASH_GROUPNORM_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.group_norm_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_GROUPNORM_KERNEL_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.group_norm_"#SRC0_TYPE"to"#DST_TYPE"_2D")

// Add kernel hashtable here
// sum sqr
#define HASH_GROUPNORM_SUM_SQR_KEY(_input0_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_output_type << 16) | (_reshape_flag << 8))

#define TENSOR_GROUPNORM_SUM_SQR_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_SUM_SQR_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_GROUPNORM_SUM_SQR_KERNEL_NAME(IN0_TYPE), \
        SOURCE },

#define TENSOR_GROUPNORM_SUM_SQR_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_SUM_SQR_KEY(IN0_TYPE, OUT_TYPE, 1), \
        HASH_GROUPNORM_SUM_SQR_KERNEL_2D_NAME(IN0_TYPE), \
        SOURCE },

#define HASH_GROUPNORM_MEAN_VARI_KEY(_input0_type, _output_type) \
    ((_input0_type << 24) | (_output_type << 16))

#define TENSOR_GROUPNORM_MEAN_VARI_KERNELS(SOURCE) \
    { HASH_GROUPNORM_MEAN_VARI_KEY(F32, F32), \
        HASH_GROUPNORM_MEAN_VARI_KERNEL_NAME, \
        SOURCE },

// normalization
#define HASH_GROUPNORM_KEY(_input0_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_output_type << 16) | (_reshape_flag << 8))

#define TENSOR_GROUPNORM_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_GROUPNORM_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_GROUPNORM_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_KEY(IN0_TYPE, OUT_TYPE, 1), \
        HASH_GROUPNORM_KERNEL_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _groupnorm_sum_sqr_kernel_map[] =
{
    // Register kernel here
    TENSOR_GROUPNORM_SUM_SQR_KERNELS( U8, F32, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_SUM_SQR_KERNELS_2D( U8, F32, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_SUM_SQR_KERNELS( F32, F32, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SUM_SQR_KERNELS_2D( F32, F32, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SUM_SQR_KERNELS( I32, F32, KERNEL_SOURCE_3 )
    TENSOR_GROUPNORM_SUM_SQR_KERNELS_2D( I32, F32, KERNEL_SOURCE_3 )
};

static const _kernel_map_type _groupnorm_mean_vari_kernel_map[] =
{
    // Register kernel here
    TENSOR_GROUPNORM_MEAN_VARI_KERNELS( KERNEL_SOURCE_2 )
};

static const _kernel_map_type _groupnorm_kernel_map[] =
{
    // Register kernel here
    TENSOR_GROUPNORM_KERNELS( U8, U8, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_KERNELS_2D( U8, U8, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_KERNELS( U8, F32, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_KERNELS_2D( U8, F32, KERNEL_SOURCE_1 )

    TENSOR_GROUPNORM_KERNELS( F32, F32, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_KERNELS_2D( F32, F32, KERNEL_SOURCE_2 )

    TENSOR_GROUPNORM_KERNELS( I32, I32, KERNEL_SOURCE_3 )
    TENSOR_GROUPNORM_KERNELS_2D( I32, I32, KERNEL_SOURCE_3 )
    TENSOR_GROUPNORM_KERNELS( I32, F32, KERNEL_SOURCE_3 )
    TENSOR_GROUPNORM_KERNELS_2D( I32, F32, KERNEL_SOURCE_3 )
};

/*
 * Kernel params
 */
static vx_param_description_t _groupnorm_sum_sqr_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GROUPNORM_SUM_SQR_PARAM_NUM  _cnt_of_array( _groupnorm_sum_sqr_kernel_param_def )

static vx_param_description_t _groupnorm_mean_vari_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GROUPNORM_MEAN_VARI_PARAM_NUM  _cnt_of_array( _groupnorm_mean_vari_kernel_param_def )

static vx_param_description_t _groupnorm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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
    // Add kererl parameters here
};
#define _GROUPNORM_PARAM_NUM  _cnt_of_array( _groupnorm_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_groupnorm_sum_sqr_initializer)
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
    vsi_int_array_t * input_shape = NULL;
    int32_t width = 0;
    int32_t chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    input_shape  = attr[0]->shape;
    width = input_shape->data[0];
    chn = attr[1]->shape->data[1];

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.local_size[0]  = 16;
    gpu_param.local_size[1]  = 1;
    gpu_param.local_size[2]  = 1;
    gpu_param.global_size[0]   = (width + 15) / 16 * 16;
    gpu_param.global_size[1]   = chn;
    gpu_param.global_size[2]   = 1;

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
} /* _group_normalization_sum_sqr_initializer() */

DEF_KERNEL_INITIALIZER(_groupnorm_mean_vari_initializer)
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
    int32_t chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    chn = attr[0]->shape->data[1];

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.local_size[0]  = 16;
    gpu_param.local_size[1]  = 1;
    gpu_param.local_size[2]  = 1;
    gpu_param.global_size[0]   = 16;
    gpu_param.global_size[1]   = chn;
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
} /* _group_normalization_sum_sqr_initializer() */

DEF_KERNEL_INITIALIZER(_groupnorm_initializer)
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
    vsi_int_array_t * input_shape = NULL;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;
    int32_t is2D = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &is2D);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input_shape  = attr[0]->shape;
    width = input_shape->data[0];
    height = input_shape->data[1];
    chn = attr[1]->shape->data[1];

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.local_size[0]  = 16;
    gpu_param.local_size[1]  = 1;
    gpu_param.local_size[2]  = 1;
    gpu_param.global_size[0]   = (width + 15) / 16 * 16;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = chn;
    if (is2D)
    {
        gpu_param.global_size[0]   = (width + 15) / 16 * 16;
        gpu_param.global_size[1]   = chn;
        gpu_param.global_size[2]   = 1;
    }

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
} /* _groupnorm_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    const uint32_t hashkey,
    _internal_kernel_e kernel_id
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vx_kernel_initialize_f  initializer = NULL;
    vx_param_description_t * param_def = NULL;
    const _kernel_map_type* kernel_map;
    size_t kernel_map_size = 0;
    size_t param_size = 0;
    uint32_t i = 0;

    switch( kernel_id )
    {
        case INTERNAL_KERNEL_SUM_SQR:
            initializer = _groupnorm_sum_sqr_initializer;
            kernel_map = _groupnorm_sum_sqr_kernel_map;
            kernel_map_size = _cnt_of_array( _groupnorm_sum_sqr_kernel_map );
            param_def = _groupnorm_sum_sqr_kernel_param_def;
            param_size = _GROUPNORM_SUM_SQR_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_MEAN_VARI:
            initializer = _groupnorm_mean_vari_initializer;
            kernel_map = _groupnorm_mean_vari_kernel_map;
            kernel_map_size = _cnt_of_array( _groupnorm_mean_vari_kernel_map );
            param_def = _groupnorm_mean_vari_kernel_param_def;
            param_size = _GROUPNORM_MEAN_VARI_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_NORM:
            initializer = _groupnorm_initializer;
            kernel_map = _groupnorm_kernel_map;
            kernel_map_size = _cnt_of_array( _groupnorm_kernel_map );
            param_def = _groupnorm_kernel_param_def;
            param_size = _GROUPNORM_PARAM_NUM;
            break;
        default:
            VSI_ASSERT( FALSE );
            return VSI_FAILURE;
    }

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == hashkey )
        {
            break;
        }
    }
    if ( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_size;
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

static int32_t _optimize_gn_shape_cl
    (
    vsi_nn_tensor_t ** inputs,
    int32_t group_size,
    int32_t group_num,
    int32_t* opt_shape,
    int32_t* is2D_flg
    )
{
    vsi_status status = VSI_SUCCESS;
    int32_t group_shape[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t new_rank = 0;
    group_shape[0] = inputs[0]->attr.size[0];
    group_shape[1] = inputs[0]->attr.size[1];
    group_shape[2] = group_size;

    vsi_nn_kernel_optimize_element_shape(group_shape, 3, opt_shape, &new_rank );

    if (opt_shape[1] == 1)
    {
        opt_shape[1] = group_num;
        opt_shape[2] = 1;
        opt_shape[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
        is2D_flg[0] = 1;
    }
    else if (new_rank == 2)
    {
        opt_shape[2] = group_num;
        opt_shape[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    }
    else
    {
        status = VSI_FAILURE;
    }

    return status;
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
#define INTERNAL_KERNEL_SIZE    (2)
#define SUM_SQR_INDEX           (0)
#define MEAN_VARI_INDEX         (1)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t sum_sqr_node_params[_GROUPNORM_SUM_SQR_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t mean_vari_node_params[_GROUPNORM_MEAN_VARI_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t node_params[_GROUPNORM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t tmp_node = NULL, tmp_node1 = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_dtype_e in0_dtype = U8;
    vsi_nn_kernel_dtype_e out_dtype = U8;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * ikernels[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * tensors[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    int32_t new_shape[VSI_NN_MAX_DIM_NUM] = { 1, 1, 1, 1 };
    int32_t is2D_flg = 0;
    uint32_t hashkeys[INTERNAL_KERNEL_SIZE] = { 0 };
    uint32_t hashkey = 0;
    int32_t i = 0;
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" );
    int32_t group_num  = vsi_nn_kernel_param_get_int32( params, "group_num" );
    int32_t group_size  = inputs[0]->attr.size[2] / group_num;

    int32_t width = inputs[0]->attr.size[0];
    int32_t height = inputs[0]->attr.size[1];
    int32_t group_stride = 1;
    float input_zp = 0;
    float input_scale = 1.0f;
    int32_t input_fl = 0;
    float output_zp = 0;
    float output_scale = 1.0f;
    int32_t output_fl = 0;
    float rSpaceOrg = 1.0f / (width * height);
    float group_ratio = 1.0f / (inputs[0]->attr.size[0] * inputs[0]->attr.size[1] * group_size);

    if ( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _optimize_gn_shape_cl(inputs, group_size, group_num, new_shape, &is2D_flg);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape, 4);
    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape, 4);

    width = new_shape[0];
    height = is2D_flg > 0 ? 1 : new_shape[1];
    group_stride = ((width + 15) / 16) * 4;

    if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        input_zp = (float)inputs[0]->attr.dtype.zero_point;
        input_scale = inputs[0]->attr.dtype.scale;
    }
    else if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        input_fl = inputs[0]->attr.dtype.fl;
        if (input_fl > 0)
        {
            input_scale = (1.0f / ((float) ((int64_t)1 << input_fl)));
        }
        else
        {
            input_scale = ((float) ((int64_t)1 << -input_fl));
        }
        input_zp = 0.0f;
    }

    if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        output_zp = (float)outputs[0]->attr.dtype.zero_point;
        output_scale = 1.0f / outputs[0]->attr.dtype.scale;
    }
    else if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        output_fl = outputs[0]->attr.dtype.fl;
        if (output_fl > 0)
        {
            output_scale = (float)((int64_t)1 << output_fl);
        }
        else
        {
            output_scale = (1.0f / (float)((int64_t)1 << -output_fl));
        }
        output_zp = 0.0f;
    }

    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        ikernels[i] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_CL );
        // Assign unique_id
        ikernels[i]->unique_id = kernel->unique_id;
    }

    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = ((new_shape[0] + 15) / 16) * 4;
    attr.size[1] = group_num;
    attr.size[2] = 1;
    attr.size[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    attr.dim_num = 4;
    tensors[SUM_SQR_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    attr.size[0] = 4;
    tensors[MEAN_VARI_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    if (in0_dtype == F16)
    {
        in0_dtype = F32;
    }
    if (out_dtype == F16)
    {
        out_dtype = F32;
    }

    hashkeys[SUM_SQR_INDEX]= HASH_GROUPNORM_SUM_SQR_KEY( in0_dtype, F32, is2D_flg );
    hashkeys[MEAN_VARI_INDEX]= HASH_GROUPNORM_MEAN_VARI_KEY( F32, F32 );
    hashkey = HASH_GROUPNORM_KEY( in0_dtype, out_dtype, is2D_flg );

    status = _query_kernel( ikernels[SUM_SQR_INDEX], hashkeys[SUM_SQR_INDEX], INTERNAL_KERNEL_SUM_SQR );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( ikernels[MEAN_VARI_INDEX], hashkeys[MEAN_VARI_INDEX], INTERNAL_KERNEL_MEAN_VARI );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( kernel, hashkey, INTERNAL_KERNEL_NORM );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    // Sum Sqr
    tmp_node = vsi_nn_kernel_create_node( graph, ikernels[SUM_SQR_INDEX] );
    if (tmp_node)
    {
        uint32_t index = 0;
        sum_sqr_node_params[index++] = rs_input;
        sum_sqr_node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[SUM_SQR_INDEX]->t;
        sum_sqr_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
        sum_sqr_node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &is2D_flg );
        sum_sqr_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_zp );
        sum_sqr_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
        sum_sqr_node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
        sum_sqr_node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &height );

        status  = vsi_nn_kernel_node_pass_param( tmp_node, sum_sqr_node_params,
            _GROUPNORM_SUM_SQR_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &sum_sqr_node_params[2] );
        vsi_nn_kernel_scalar_release( &sum_sqr_node_params[3] );
        vsi_nn_kernel_scalar_release( &sum_sqr_node_params[4] );
        vsi_nn_kernel_scalar_release( &sum_sqr_node_params[5] );
        vsi_nn_kernel_scalar_release( &sum_sqr_node_params[6] );
        vsi_nn_kernel_scalar_release( &sum_sqr_node_params[7] );
        vsi_nn_kernel_node_release( &tmp_node );
    }

    // mean vari
    tmp_node1 = vsi_nn_kernel_create_node( graph, ikernels[MEAN_VARI_INDEX] );
    if (tmp_node1)
    {
        uint32_t index = 0;
        mean_vari_node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[SUM_SQR_INDEX]->t;
        mean_vari_node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[MEAN_VARI_INDEX]->t;
        mean_vari_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
        mean_vari_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &group_ratio );
        mean_vari_node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &group_stride );

        status  = vsi_nn_kernel_node_pass_param( tmp_node1, mean_vari_node_params,
            _GROUPNORM_MEAN_VARI_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &mean_vari_node_params[2] );
        vsi_nn_kernel_scalar_release( &mean_vari_node_params[3] );
        vsi_nn_kernel_scalar_release( &mean_vari_node_params[4] );
        vsi_nn_kernel_node_release( &tmp_node1 );
    }

    // Nomalization
    node = vsi_nn_kernel_create_node( graph, kernel );
    if (node)
    {
        uint32_t index = 0;
        int32_t  pStride = 0;
        if (!is2D_flg)
        {
            pStride = inputs[1]->attr.size[0] / new_shape[1];
            rSpaceOrg = 1.0f / (new_shape[0] / pStride);
        }
        node_params[index++] = rs_input;
        node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
        node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
        node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[MEAN_VARI_INDEX]->t;
        node_params[index++] = rs_output;
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &is2D_flg );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_zp );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &output_zp );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &output_scale );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &rSpaceOrg );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &height );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pStride );

        status  = vsi_nn_kernel_node_pass_param( node, node_params,
            _GROUPNORM_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &node_params[5] );
        vsi_nn_kernel_scalar_release( &node_params[6] );
        vsi_nn_kernel_scalar_release( &node_params[7] );
        vsi_nn_kernel_scalar_release( &node_params[8] );
        vsi_nn_kernel_scalar_release( &node_params[9] );
        vsi_nn_kernel_scalar_release( &node_params[10] );
        vsi_nn_kernel_scalar_release( &node_params[11] );
        vsi_nn_kernel_scalar_release( &node_params[12] );
        vsi_nn_kernel_scalar_release( &node_params[13] );
        vsi_nn_kernel_scalar_release( &node_params[14] );
    }

    /* Pass parameters to node. */
final:
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        if ( ikernels[i] )
        {
            vsi_nn_kernel_release( &ikernels[i] );
        }
        if ( tensors[i] )
        {
            vsi_nn_ReleaseTensor( &tensors[i] );
        }
    }
#undef INTERNAL_KERNEL_SIZE
#undef SUM_SQR_INDEX
#undef MEAN_VARI_INDEX
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( group_norm, _setup )

