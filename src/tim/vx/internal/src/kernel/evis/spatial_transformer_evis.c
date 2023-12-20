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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

typedef enum
{
    INTERNAL_KERNEL_GET_MATRIX,
    INTERNAL_KERNEL_WARP_AFFINE,
} _internal_kernel_e;

#define _GET_MATRIX_SOURCE        "get_matrix"
#define _WARP_AFFINE_SOURCE       "warp_affine"

// Add kernel hashtable here
#define GET_MATRIX_HASH_KEY( IN1_DTYPE, OUT_DTYPE ) \
        (( IN1_DTYPE << 8 ) | ( OUT_DTYPE ))
#define GET_MATRIX_KERNEL_MAP( IN1_DTYPE, OUT_DTYPE ) \
        { GET_MATRIX_HASH_KEY( IN1_DTYPE, OUT_DTYPE ), \
        CVIVANTE_NAMESPACE("evis.get_matrix_"#IN1_DTYPE"toF32"), \
        _GET_MATRIX_SOURCE }

#define WARP_AFFINE_HASH_KEY( IN0_DTYPE, OUT_DTYPE ) \
        (( IN0_DTYPE << 8 ) | ( OUT_DTYPE ))
#define WARP_AFFINE_KERNEL_MAP( IN0_DTYPE, OUT_DTYPE ) \
        { WARP_AFFINE_HASH_KEY( IN0_DTYPE, OUT_DTYPE ), \
          CVIVANTE_NAMESPACE("evis.warp_affine_"#IN0_DTYPE"to"#OUT_DTYPE), \
          _WARP_AFFINE_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _get_matrix_kernel_map[] =
{
    // Register kernel here
    GET_MATRIX_KERNEL_MAP( F16, F32 ),
    GET_MATRIX_KERNEL_MAP( I16, F32 ),
    GET_MATRIX_KERNEL_MAP( U8,  F32 ),
    GET_MATRIX_KERNEL_MAP( I8,  F32 ),
};

static const _kernel_map_type _warp_affine_kernel_map[] =
{
    // Register kernel here
    WARP_AFFINE_KERNEL_MAP( F16, F16 ),
    WARP_AFFINE_KERNEL_MAP( U8,  U8 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _get_matrix_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GET_MATRIX_PARAM_NUM  _cnt_of_array( _get_matrix_kernel_param_def )
#define HAS_THETA_1_1   (2)
#define HAS_THETA_1_2   (3)
#define HAS_THETA_1_3   (4)
#define HAS_THETA_2_1   (5)
#define HAS_THETA_2_2   (6)
#define HAS_THETA_2_3   (7)
#define THETA_1_1       (8)
#define THETA_1_2       (9)
#define THETA_1_3       (10)
#define THETA_2_1       (11)
#define THETA_2_2       (12)
#define THETA_2_3       (13)
#define I_WIDTH         (14)
#define I_HEIGHT        (15)
#define O_WIDTH         (16)
#define O_HEIGHT        (17)

static vx_param_description_t _warp_affine_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _WARP_AFFINE_PARAM_NUM  _cnt_of_array( _warp_affine_kernel_param_def )
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_get_matrix_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
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
    vsi_size_array_t * in_shape          = NULL;
    float    theta[8] = {0};
    float    input_scale = 1.0f;
    float    input_tail  = 0;
    float    input_w = 1.0f;
    float    input_h = 1.0f;
    float    output_w = 1.0f;
    float    output_h = 1.0f;
    float    scale[4] = {0};

    VSI_UNREFERENCED(param_size);

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr buffer fail.", final );

    if ( attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr->dfp.fl;
        if (fl > 0)
        {
            input_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if ( attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input_scale  = attr->asymm.scale;
        input_tail = 0 - attr->asymm.zero_point * input_scale;
    }

    in_shape  = attr->shape;

    status  = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_1_1], &theta[0]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_1_2], &theta[1]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_1_3], &theta[2]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_2_1], &theta[4]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_2_2], &theta[5]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[THETA_2_3], &theta[6]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[I_WIDTH], &input_w);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[I_HEIGHT], &input_h);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[O_WIDTH], &output_w);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[O_HEIGHT], &output_h);
    CHECK_STATUS_FAIL_GOTO( status, final );

    scale[0] = input_w / output_w;
    scale[1] = input_h / output_h;
    scale[2] = input_w / output_h;
    scale[3] = input_h / output_w;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0] = 1;
    gpu_param.global_size[1] = in_shape->data[1];

    status = vsi_nn_kernel_gpu_add_param( node,
        "theta_1", &theta[0] );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "theta_2", &theta[4] );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "scale", &scale );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "input_scale", &input_scale );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "input_tail", &input_tail );
    CHECK_STATUS_FAIL_GOTO(status, final );

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (attr)
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
        attr = NULL;
    }

    return status;
} /* _get_matrix_initializer() */

DEF_KERNEL_INITIALIZER(_warp_affine_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr[2]  = {NULL};
    vsi_size_array_t * out_shape          = NULL;
    float    input_scale = 1.0f;
    float    input_tail  = 0;
    float    output_scale = 1.0f;
    float    output_zp  = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[0]->dfp.fl;
        if (fl > 0)
        {
            input_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input_scale  = attr[0]->asymm.scale;
        input_tail = 0 - attr[0]->asymm.zero_point * input_scale;
    }

    if (attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        int32_t fl = attr[1]->dfp.fl;

        if (fl >= 0)
        {
            output_scale = (vx_float32) ((vx_int64)1 << fl);
        }
        else if (fl < 0)
        {
            output_scale = 1.0f / (vx_float32) ((vx_int64)1 << -fl);
        }
    }
    else if (attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        output_scale   = 1.0f / attr[1]->asymm.scale;
        output_zp = (float)attr[1]->asymm.zero_point;
    }

    out_shape  = attr[1]->shape;

    gpu_param.global_scale[0] = 2;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = out_shape->data[1];
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    {
        gpu_dp_inst_t uniConvertDatatoF32_0_4x4 = {{
            0x01010101, // TCfg
            0x01010000, // ASelt
            0x00010000, 0x00010000, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertDatatoF32_1_4x4 = {{
            0x01010101, // TCfg
            0x01010000, // ASelt
            0x00030002, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertDatatoF32_0_4x4", &uniConvertDatatoF32_0_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniConvertDatatoF32_1_4x4", &uniConvertDatatoF32_1_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "input_scale", &input_scale);
        status |= vsi_nn_kernel_gpu_add_param( node, "input_tail", &input_tail);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_scale", &output_scale);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_zp", &output_zp);
        if (attr[1]->dtype == F16)
        {
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniExtract8Data_2x8", &uniExtractHalf8_2x8 );
        }
        else
        {
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
        }
        CHECK_STATUS_FAIL_GOTO(status, final );
    }


    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
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
} /* _warp_affine_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    const uint32_t hashkey,
    _internal_kernel_e kernel_id
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
        case INTERNAL_KERNEL_GET_MATRIX:
            initializer = _get_matrix_initializer;
            kernel_map = _get_matrix_kernel_map;
            kernel_map_size = _cnt_of_array( _get_matrix_kernel_map );
            param_def = _get_matrix_kernel_param_def;
            param_size = _GET_MATRIX_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_WARP_AFFINE:
            initializer = _warp_affine_initializer;
            kernel_map = _warp_affine_kernel_map;
            kernel_map_size = _cnt_of_array( _warp_affine_kernel_map );
            param_def = _warp_affine_kernel_param_def;
            param_size = _WARP_AFFINE_PARAM_NUM;
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
#define INTERNAL_KERNEL_SIZE    (2)
#define MATRIX_INDEX  (0)
#define WARP_AFFINE_INDEX   (1)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_GET_MATRIX_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t warp_affine_node_params[_WARP_AFFINE_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * ikernels[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * tensors[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * warp_affine_tensors[2] = {NULL};
    uint32_t hashkeys[INTERNAL_KERNEL_SIZE] = { 0 };
    int32_t has_theta_1_1  = vsi_nn_kernel_param_get_int32( params, "has_theta_1_1" );
    int32_t has_theta_1_2  = vsi_nn_kernel_param_get_int32( params, "has_theta_1_2" );
    int32_t has_theta_1_3  = vsi_nn_kernel_param_get_int32( params, "has_theta_1_3" );
    int32_t has_theta_2_1  = vsi_nn_kernel_param_get_int32( params, "has_theta_2_1" );
    int32_t has_theta_2_2  = vsi_nn_kernel_param_get_int32( params, "has_theta_2_2" );
    int32_t has_theta_2_3  = vsi_nn_kernel_param_get_int32( params, "has_theta_2_3" );
    float theta_1_1  = vsi_nn_kernel_param_get_float32( params, "theta_1_1" );
    float theta_1_2  = vsi_nn_kernel_param_get_float32( params, "theta_1_2" );
    float theta_1_3  = vsi_nn_kernel_param_get_float32( params, "theta_1_3" );
    float theta_2_1  = vsi_nn_kernel_param_get_float32( params, "theta_2_1" );
    float theta_2_2  = vsi_nn_kernel_param_get_float32( params, "theta_2_2" );
    float theta_2_3  = vsi_nn_kernel_param_get_float32( params, "theta_2_3" );
    int32_t align_corners  = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    float input_w    = (float)inputs[0]->attr.size[0];
    float input_h    = (float)inputs[0]->attr.size[1];
    float output_w   = (float)outputs[0]->attr.size[0];
    float output_h   = (float)outputs[0]->attr.size[1];
    int32_t i = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if (align_corners && output_w > 1)
    {
        output_w = output_w - 1;
    }

    if (align_corners && output_h > 1)
    {
        output_h = output_h - 1;
    }

    // Check if gpu can support the size
    if( !vsi_nn_kernel_gpu_check_shape(
        outputs[0]->attr.size, outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        ikernels[i] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        // Assign unique_id
        ikernels[i]->unique_id = kernel->unique_id;
    }

    memcpy( &attr, &(inputs[1]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.size[0] = 16;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT16;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    tensors[0] = vsi_nn_CreateTensor( graph, &attr );

    attr.size[3] = attr.size[1];
    attr.size[2] = attr.size[1] = 1;
    attr.dim_num = inputs[0]->attr.dim_num;
    tensors[1] = vsi_nn_reshape_tensor( graph,
                tensors[0], attr.size, attr.dim_num );

    warp_affine_tensors[0] = inputs[0];
    warp_affine_tensors[1] = tensors[1];

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    hashkeys[MATRIX_INDEX]= GET_MATRIX_HASH_KEY( in1_dtype, F32 );
    hashkeys[WARP_AFFINE_INDEX] = WARP_AFFINE_HASH_KEY( in0_dtype, out_dtype );

    status = _query_kernel( ikernels[MATRIX_INDEX], hashkeys[MATRIX_INDEX], INTERNAL_KERNEL_GET_MATRIX );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }

    status = _query_kernel( ikernels[WARP_AFFINE_INDEX], hashkeys[WARP_AFFINE_INDEX], INTERNAL_KERNEL_WARP_AFFINE );
    if( VSI_SUCCESS != status )
    {
        goto final;
    }

    // Get Matrix
    node = vsi_nn_kernel_create_node( graph, ikernels[MATRIX_INDEX] );

    if (node)
    {
        vsi_nn_kernel_node_pack_io( node_params, _GET_MATRIX_PARAM_NUM,
                &inputs[1], 1, &tensors[0], 1 );
        node_params[HAS_THETA_1_1] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_1_1 );
        node_params[HAS_THETA_1_2] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_1_2 );
        node_params[HAS_THETA_1_3] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_1_3 );
        node_params[HAS_THETA_2_1] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_2_1 );
        node_params[HAS_THETA_2_2] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_2_2 );
        node_params[HAS_THETA_2_3] = vsi_nn_kernel_scalar_create( graph, I32, &has_theta_2_3 );
        node_params[THETA_1_1] = vsi_nn_kernel_scalar_create( graph, F32, &theta_1_1 );
        node_params[THETA_1_2] = vsi_nn_kernel_scalar_create( graph, F32, &theta_1_2 );
        node_params[THETA_1_3] = vsi_nn_kernel_scalar_create( graph, F32, &theta_1_3 );
        node_params[THETA_2_1] = vsi_nn_kernel_scalar_create( graph, F32, &theta_2_1 );
        node_params[THETA_2_2] = vsi_nn_kernel_scalar_create( graph, F32, &theta_2_2 );
        node_params[THETA_2_3] = vsi_nn_kernel_scalar_create( graph, F32, &theta_2_3 );
        node_params[I_WIDTH] = vsi_nn_kernel_scalar_create( graph, F32, &input_w );
        node_params[I_HEIGHT] = vsi_nn_kernel_scalar_create( graph, F32, &input_h );
        node_params[O_WIDTH] = vsi_nn_kernel_scalar_create( graph, F32, &output_w );
        node_params[O_HEIGHT] = vsi_nn_kernel_scalar_create( graph, F32, &output_h );
        status  = vsi_nn_kernel_node_pass_param( node, node_params, _GET_MATRIX_PARAM_NUM );
        vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_1_1] );
        vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_1_2] );
        vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_1_3] );
        vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_2_1] );
        vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_2_2] );
        vsi_nn_kernel_scalar_release( &node_params[HAS_THETA_2_3] );
        vsi_nn_kernel_scalar_release( &node_params[THETA_1_1] );
        vsi_nn_kernel_scalar_release( &node_params[THETA_1_2] );
        vsi_nn_kernel_scalar_release( &node_params[THETA_1_3] );
        vsi_nn_kernel_scalar_release( &node_params[THETA_2_1] );
        vsi_nn_kernel_scalar_release( &node_params[THETA_2_2] );
        vsi_nn_kernel_scalar_release( &node_params[THETA_2_3] );
        vsi_nn_kernel_scalar_release( &node_params[I_WIDTH] );
        vsi_nn_kernel_scalar_release( &node_params[I_HEIGHT] );
        vsi_nn_kernel_scalar_release( &node_params[O_WIDTH] );
        vsi_nn_kernel_scalar_release( &node_params[O_HEIGHT] );
        vsi_nn_kernel_node_release( &node );
    }

    // Warp Affine
    node = vsi_nn_kernel_create_node( graph, ikernels[WARP_AFFINE_INDEX] );
    if (node)
    {
        vx_border_t border;
        border.mode = VX_BORDER_CONSTANT;
        border.constant_value.U32 = 0;
        border.constant_value.S16 = 0;
        border.constant_value.U8 = 0;
        if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8 &&
            inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
        }
        status  = vsi_nn_kernel_node_set_border( node, &border );
        if ( VSI_SUCCESS != status )
        {
            goto final;
        }
        vsi_nn_kernel_node_pack_io( warp_affine_node_params, _WARP_AFFINE_PARAM_NUM,
                warp_affine_tensors, 2, outputs, 1 );
        status  = vsi_nn_kernel_node_pass_param( node, warp_affine_node_params, _WARP_AFFINE_PARAM_NUM );
        if ( VSI_SUCCESS != status )
        {
            goto final;
        }
    }
final:
    for ( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
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

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( spatial_transformer, _setup )
