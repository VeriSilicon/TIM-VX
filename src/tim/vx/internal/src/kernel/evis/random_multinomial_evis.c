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
#include <math.h>
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

typedef enum
{
    INTERNAL_KERNEL_SEED,
    INTERNAL_KERNEL_CDF,
    INTERNAL_KERNEL_MULTINOMIAL,
} _internal_kernel_e;

/*
 * Define kernel meta.
 */
#define _MULTINOMIAL_KERNEL_SOURCE  "random_multinomial"
#define _MULTINOMIAL_KERNEL_NAME    CVIVANTE_NAMESPACE("evis.random_multinomial")
#define _CDF_KERNEL_SOURCE          "random_multinomial"
#define _SEED_KERNEL_SOURCE         "random_multinomial"
#define _SEED_KERNEL_NAME           CVIVANTE_NAMESPACE("evis.random_seed")

// Add kernel hashtable here
#define MULTINOMIAL_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        ((IN0_DTYPE << 16) | ( IN1_DTYPE << 8 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
        { MULTINOMIAL_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), \
            _MULTINOMIAL_KERNEL_NAME, SOURCE }

#define CDF_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define CDF_PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { CDF_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
          CVIVANTE_NAMESPACE("evis.random_multinomial_cdf_"#IN_DTYPE), \
          SOURCE }

#define SEED_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define SEED_PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { SEED_HASH_KEY( IN_DTYPE, OUT_DTYPE ), _SEED_KERNEL_NAME, SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _seed_kernel_map[] =
{
    // Register kernel here
    SEED_PACK_KERNEL_MAP( I32, F32, _SEED_KERNEL_SOURCE ),
};

static const _kernel_map_type _cdf_kernel_map[] =
{
    // Register kernel here
    CDF_PACK_KERNEL_MAP( F16, F32, _CDF_KERNEL_SOURCE ),
    CDF_PACK_KERNEL_MAP( F32, F32, _CDF_KERNEL_SOURCE ),
};

static const _kernel_map_type _kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, I32, _MULTINOMIAL_KERNEL_SOURCE ),
};

/*
 * Kernel params
 */
static vx_param_description_t _kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define SCALAR_CLASS_SIZE   (3)
#define _PARAM_NUM  _cnt_of_array( _kernel_param_def )

static vx_param_description_t _cdf_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CDF_PARAM_NUM  _cnt_of_array( _cdf_kernel_param_def )

static vx_param_description_t _seed_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SEED_PARAM_NUM  _cnt_of_array( _seed_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_multinomial_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * attr  = NULL;
    vsi_int_array_t * in_shape          = NULL;

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr buffer fail.", final );

    in_shape  = attr->shape;

    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (in_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = in_shape->data[1];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (attr)
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
        attr = NULL;
    }

    return status;
} /* _multinomial_initializer() */

DEF_KERNEL_INITIALIZER(_cdf_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * attr  = NULL;
    vsi_int_array_t * in_shape          = NULL;
    uint32_t      class_max_iter        = 0;
    uint32_t      class_size            = 0;
    uint32_t      batch                 = 0;

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr buffer fail.", final );

    in_shape  = attr->shape;

    class_size = in_shape->data[0];
    batch = in_shape->data[1];
    if (attr->dtype == F32)
    {
        class_max_iter = (class_size + 3) >> 2;
    }
    else
    {
        class_max_iter = (class_size + 7) >> 3;
    }

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0]  = 1;
    gpu_param.global_size[1]  = batch;


    if (attr->dtype == F16)
    {
        gpu_dp_inst_t uniPackMaxData_2x8 = {{
            0x00000111, // TCfg
            0x00000000, // ASelt
            0x00050300, 0x00000000, // ABin
            0x00000222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00004400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGetSubData0to3_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x05050505, // BSelt
            0x00110011, 0x00110011, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGetSubData4to7_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        status  = vsi_nn_kernel_gpu_add_param(node, "uniPackMaxData_2x8", &uniPackMaxData_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniGetSubData0to3_4x4", &uniGetSubData0to3_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniGetSubData4to7_4x4", &uniGetSubData4to7_4x4);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status  = vsi_nn_kernel_gpu_add_param(node, "class_max_iter", &class_max_iter);
    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (attr)
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
        attr = NULL;
    }

    return status;
} /* _cdf_initializer() */

DEF_KERNEL_INITIALIZER(_seed_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * attr  = NULL;
    vsi_int_array_t * out_shape         = NULL;
    uint32_t          stride            = 0;
    uint32_t          iter              = 8;
    float             rand_max          = (float)(pow(2.0,32));
    float             re_rand_max       = 1 / rand_max;

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr buffer fail.", final );

    out_shape  = attr->shape;
    iter = (out_shape->data[0] + 3) / 4;

    stride = iter * 4;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0]  = 1;
    gpu_param.global_size[1]  = 1;

    status  = vsi_nn_kernel_gpu_add_param(node, "stride", &stride);
    status |= vsi_nn_kernel_gpu_add_param(node, "iter", &iter);
    status |= vsi_nn_kernel_gpu_add_param(node, "re_rand_max", &re_rand_max);
    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (attr)
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
        attr = NULL;
    }

    return status;
} /* _seed_initializer() */

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
    vx_kernel_initialize_f  initializer = NULL;
    vx_param_description_t * param_def;
    vsi_status status = VSI_FAILURE;
    const _kernel_map_type* kernel_map;
    size_t kernel_map_size;
    size_t param_size;
    uint32_t i;

    switch( kernel_id )
    {
        case INTERNAL_KERNEL_SEED:
            initializer = _seed_initializer;
            kernel_map = _seed_kernel_map;
            kernel_map_size = _cnt_of_array( _seed_kernel_map );
            param_def = _seed_kernel_param_def;
            param_size = _SEED_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_CDF:
            initializer = _cdf_initializer;
            kernel_map = _cdf_kernel_map;
            kernel_map_size = _cnt_of_array( _cdf_kernel_map );
            param_def = _cdf_kernel_param_def;
            param_size = _CDF_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_MULTINOMIAL:
            initializer = _multinomial_initializer;
            kernel_map = _kernel_map;
            kernel_map_size = _cnt_of_array( _kernel_map );
            param_def = _kernel_param_def;
            param_size = _PARAM_NUM;
            break;
        default:
            VSI_ASSERT( FALSE );
            return VSI_FAILURE;
    }

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == hashkey )
        {
            break;
        }
    }
    if( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
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
#define INTERNAL_KERNEL_SIZE    (3)
#define SEED_INDEX  (0)
#define CDF_INDEX   (1)
#define SEEDS_INDEX  (2)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t cdf_node_params[_CDF_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t seed_node_params[_SEED_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * ikernels[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * tensors[INTERNAL_KERNEL_SIZE] = { NULL };
    int32_t class_max_stride = 0;
    int32_t class_size = 0;
    uint32_t hashkeys[INTERNAL_KERNEL_SIZE] = { 0 };
    uint32_t hashkey = 0;
    int32_t i;

    // Check if gpu can support the size
    if( !vsi_nn_kernel_gpu_check_shape(
        (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        ikernels[i] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        // Assign unique_id
        ikernels[i]->unique_id = kernel->unique_id;
    }
    if( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        class_max_stride = (int32_t)gpu_align_p2(inputs[0]->attr.size[0], 4);
    }
    else
    {
        class_max_stride = (int32_t)gpu_align_p2(inputs[0]->attr.size[0], 8);
    }
    class_size = inputs[0]->attr.size[0];

    memcpy( &attr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    tensors[SEED_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    attr.size[0] = class_max_stride * inputs[0]->attr.size[1];
    attr.size[1] = inputs[0]->attr.size[1];
    attr.dim_num = 2;
    tensors[CDF_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    memcpy( &attr, &(inputs[1]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.size[1] = 1;
    attr.dim_num = 2;
    tensors[SEEDS_INDEX] = vsi_nn_reshape_tensor( graph,
                inputs[1], (uint32_t*)attr.size, attr.dim_num );

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    hashkeys[SEED_INDEX]= SEED_HASH_KEY( in1_dtype, F32 );
    hashkeys[CDF_INDEX] = CDF_HASH_KEY( in0_dtype, F32 );
    hashkey = MULTINOMIAL_HASH_KEY( F32, F32, out_dtype );

    status = _query_kernel( ikernels[SEED_INDEX], hashkeys[SEED_INDEX], INTERNAL_KERNEL_SEED );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( ikernels[CDF_INDEX], hashkeys[CDF_INDEX], INTERNAL_KERNEL_CDF );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( kernel, hashkey, INTERNAL_KERNEL_MULTINOMIAL );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }

    // Seed
    node = vsi_nn_kernel_create_node( graph, ikernels[SEED_INDEX] );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( seed_node_params, _SEED_PARAM_NUM,
            &tensors[SEEDS_INDEX], 1, &tensors[SEED_INDEX], 1 );
    status  = vsi_nn_kernel_node_pass_param( node, seed_node_params, _SEED_PARAM_NUM );
    VSI_ASSERT( status == VSI_SUCCESS );
    vsi_nn_kernel_node_release( &node );

    // CDF
    node = vsi_nn_kernel_create_node( graph, ikernels[CDF_INDEX] );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( cdf_node_params, _CDF_PARAM_NUM,
            &inputs[0], 1, &tensors[CDF_INDEX], 1 );
    status  = vsi_nn_kernel_node_pass_param( node, cdf_node_params, _CDF_PARAM_NUM );
    VSI_ASSERT( status == VSI_SUCCESS );
    vsi_nn_kernel_node_release( &node );

    // Multinomial
    node = vsi_nn_kernel_create_node( graph, kernel );
    VSI_ASSERT( node != NULL );
    vsi_nn_kernel_node_pack_io( node_params, _PARAM_NUM, tensors, 2, outputs, 1 );
    node_params[SCALAR_CLASS_SIZE] = vsi_nn_kernel_scalar_create( graph, I32, &class_size );
    status  = vsi_nn_kernel_node_pass_param( node, node_params, _PARAM_NUM );
    VSI_ASSERT( status == VSI_SUCCESS );
    vsi_nn_kernel_scalar_release( &node_params[SCALAR_CLASS_SIZE] );

    /* Pass parameters to node. */
final:
    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        if( ikernels[i] )
        {
            vsi_nn_kernel_release( &ikernels[i] );
        }
        if( tensors[i] )
        {
            vsi_nn_ReleaseTensor( &tensors[i] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( random_multinomial, _setup )

