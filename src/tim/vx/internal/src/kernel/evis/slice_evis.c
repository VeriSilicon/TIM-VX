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
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS


#define _SLICE_KERNEL_SOURCE      "slice"
#define _SLICE_KERNEL_NAME        CVIVANTE_NAMESPACE("evis.slice")

    // Add kernel hashtable here
#define SLICE_SH_KERNEL_NAME(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    CVIVANTE_NAMESPACE("evis.slice_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE)

    // Add kernel hashtable here
#define SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE , _IMAGE_2D, _SAMEFL) \
    (( IN1_DTYPE << 18 ) | ( IN0_DTYPE << 10 ) | ( OUT_DTYPE << 2 ) | (_IMAGE_2D << 1) | (_SAMEFL))

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
{   SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0, 0 ), \
    SLICE_SH_KERNEL_NAME( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), SOURCE }

#define SLICE_SH_KERNEL_2D_NAME(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    CVIVANTE_NAMESPACE("evis.slice_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE"_2D")

#define PACK_KERNEL_MAP_2D( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
{   SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1, 0 ), \
    SLICE_SH_KERNEL_2D_NAME( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), SOURCE }

#define SLICE_SH_KERNEL_SAMEFL_NAME(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    CVIVANTE_NAMESPACE("evis.slice_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE"_SAMEFL")

#define PACK_KERNEL_MAP_SAMEFL( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
{   SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0, 1 ), \
    SLICE_SH_KERNEL_SAMEFL_NAME( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), SOURCE }

#define SLICE_SH_KERNEL_SAMEFL_2D_NAME(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    CVIVANTE_NAMESPACE("evis.slice_"#IN0_DTYPE"_"#IN1_DTYPE"to"#OUT_DTYPE"_SAMEFL_2D")

#define PACK_KERNEL_MAP_SAMEFL_2D( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, SOURCE ) \
{   SLICE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1, 1 ), \
    SLICE_SH_KERNEL_SAMEFL_2D_NAME( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), SOURCE }

    typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _slice_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F16, I32, F16, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I16, I32, I16, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U8, I32,  U8,  _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I8, I32,  I8,  _SLICE_KERNEL_SOURCE ),

    PACK_KERNEL_MAP_2D( F16, I32, F16, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I16, I32, I16, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( U8, I32,  U8,  _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I8, I32,  I8,  _SLICE_KERNEL_SOURCE ),

    PACK_KERNEL_MAP_SAMEFL( I16, I32, I16, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_SAMEFL( U8,  I32, U8, _SLICE_KERNEL_SOURCE ),

    PACK_KERNEL_MAP_SAMEFL_2D( I16, I32, I16, _SLICE_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_SAMEFL_2D( U8, I32,  U8,  _SLICE_KERNEL_SOURCE ),
};

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
/*
* Kernel params
*/
static vx_param_description_t _slice_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _SLICE_PARAM_NUM  _cnt_of_array( _slice_kernel_param_def )
#define SCALAR_SAMLEFL_VALUE          (3)
/*
* Kernel initializer
*/
DEF_KERNEL_INITIALIZER(_slice_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
#define _PACK_SLICE_KEY( IN0_TYPE, OUT_TYPE, SAMLEFL)    \
    (IN0_TYPE | (OUT_TYPE << 8) | (SAMLEFL << 16))
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t * input_attr    = NULL;
    vsi_int_array_t * out_shape                 = NULL;
    vsi_nn_kernel_dtype_e        input_dtype    = F16;
    vsi_nn_kernel_dtype_e        output_dtype   = F16;
    float     scaleIn         = 1.0f;
    float     scaleOut        = 1.0f;
    int32_t   output_ZP       = 0;
    int32_t   input_ZP        = 0;
    int32_t   srcFixPointPos  = 0;
    int32_t   dstFixPointPos  = 0;
    int32_t   is_samefl       = 0;
    uint32_t  pack_key        = 0;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_SAMLEFL_VALUE], &is_samefl);
    CHECK_STATUS_FAIL_GOTO(status, final );

    out_shape  = output_attr->shape;
    input_dtype  = input_attr->dtype;
    output_dtype = output_attr->dtype;

    pack_key = _PACK_SLICE_KEY( input_dtype, output_dtype, is_samefl);

    if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        srcFixPointPos   = input_attr->dfp.fl;
        if (srcFixPointPos > 0)
        {
            scaleIn = (1.0f / ((float) ((int64_t)1 << srcFixPointPos)));
        }
        else
        {
            scaleIn = ((float) ((int64_t)1 << -srcFixPointPos));
        }
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant)
    {
        input_ZP         = input_attr->asymm.zero_point;
        scaleIn          = input_attr->asymm.scale;
    }

    if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        dstFixPointPos   = output_attr->dfp.fl;
        if (dstFixPointPos > 0)
        {
            scaleOut = (1.0f / ((float) ((int64_t)1 << dstFixPointPos)));
        }
        else
        {
            scaleOut = ((float) ((int64_t)1 << -dstFixPointPos));
        }
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant)
    {
        output_ZP        = output_attr->asymm.zero_point;
        scaleOut         = output_attr->asymm.scale;
    }

    if ((F16 == input_dtype)
        || (I16 == input_dtype)
        || (BF16 == input_dtype)
        )
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }
    else
    {
        gpu_param.global_scale[0]  = 16;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }

    gpu_param.dim = out_shape->size < 3 ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
        (out_shape->data[0] + gpu_param.global_scale[0] - 1)
        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
        (out_shape->data[1] + gpu_param.global_scale[1] - 1)
        / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    switch (pack_key)
    {
    case _PACK_SLICE_KEY(I16, I16, 0):
    case _PACK_SLICE_KEY(U8,  U8,  0):
    case _PACK_SLICE_KEY(I8,  I8,  0):
    case _PACK_SLICE_KEY(I16, F16, 0):
    case _PACK_SLICE_KEY(U8,  F16, 0):
    case _PACK_SLICE_KEY(I8,  F16, 0):
    case _PACK_SLICE_KEY(F16, I16, 0):
    case _PACK_SLICE_KEY(F16, U8,  0):
    case _PACK_SLICE_KEY(F16, I8,  0):
        {
            float     uint8Scale = scaleIn / scaleOut;
            uint16_t  M0                   = 0;
            int32_t   postShift            = 0;
            uint32_t  multAndoutZP[2]      = {0};
            gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniU8MulAndPostShift_Hi_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x1b1a1918, 0x1f1e1d1c, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            gpu_quantize_multiplier_16bit(uint8Scale, &M0, &postShift);
            multAndoutZP[0] = (uint32_t)(M0);
            multAndoutZP[1] = (uint32_t)((output_ZP << postShift) - input_ZP * M0);

            uniU8MulAndPostShift_Lo_2x8.data[7] |= (postShift & 0x1F);
            uniU8MulAndPostShift_Hi_2x8.data[7] |= (postShift & 0x1F);

            status  = vsi_nn_kernel_gpu_add_param( node, "uniU8MulAndPostShift_Lo_2x8", &uniU8MulAndPostShift_Lo_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniU8MulAndPostShift_Hi_2x8", &uniU8MulAndPostShift_Hi_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP", multAndoutZP);
        }
        break;
    default:
        break;
    }
    CHECK_STATUS_FAIL_GOTO(status, final );

#undef _PACK_SLICE_KEY
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    SAFE_FREE_TENSOR_ATTR(input_attr);
    return status;
} /* _slice_initializer() */

static vsi_bool _is_same_quant
    (
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_dtype_t *src_dtype = NULL,*dst_dtype = NULL;

    src_dtype = &inputs[0]->attr.dtype;
    dst_dtype = &outputs[0]->attr.dtype;

    if (vsi_nn_DtypeCompare(src_dtype, dst_dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

/*
* Query kernel
*/
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const* const inputs,
    vsi_nn_tensor_t * const* const outputs,
    vsi_bool image_2d,
    vsi_bool is_same_quant
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _slice_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _slice_kernel_map );
    vx_param_description_t * param_def  = _slice_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _slice_kernel_param_def );
    vx_kernel_initialize_f  initializer = _slice_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (is_same_quant && (F16 == in0_dtype || BF16 == in0_dtype) )
    {
        in0_dtype = I16;
        out_dtype = I16;
    }
    else if (is_same_quant && (I8 == in0_dtype || BOOL8 == in0_dtype) )
    {
        in0_dtype = U8;
        out_dtype = U8;
    }

    key = SLICE_HASH_KEY( in0_dtype, in1_dtype, out_dtype, image_2d, is_same_quant );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_SLICE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    uint32_t rank[_IO_NUM] = {0};
    int32_t  shapes[_IO_NUM][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_tensor_t* reshape_tensors[_IO_NUM] = { NULL };
    int32_t i = 0;
    int32_t input_batch = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    int32_t output_batch = outputs[0]->attr.dim_num > 3 ? outputs[0]->attr.size[3] : 1;
    vsi_bool is_same_quant = FALSE;

    vsi_nn_kernel_optimize_1d_tensor_shape( (const int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num,
        shapes[0], &rank[0]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const int32_t*)inputs[1]->attr.size, inputs[1]->attr.dim_num,
        shapes[1], &rank[1]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num,
        shapes[2], &rank[2]);

    for (i = 0; i < _INPUT_NUM; i++)
    {
        reshape_tensors[i] = vsi_nn_reshape_tensor( graph,
            inputs[i], (uint32_t*)shapes[i], rank[i] );
    }
    reshape_tensors[_INPUT_NUM] = vsi_nn_reshape_tensor( graph,
        outputs[0], (uint32_t*)shapes[_INPUT_NUM], rank[_INPUT_NUM] );

    if ( !vsi_nn_kernel_gpu_check_shape( (int32_t*)reshape_tensors[0]->attr.size,
        reshape_tensors[0]->attr.dim_num ) || input_batch != output_batch )
    {
        return NULL;
    }

    image_2d = (rank[0] < 3 || shapes[0][2] == 1);
    is_same_quant = _is_same_quant(inputs, outputs);

    status = _query_kernel( kernel, inputs, outputs , image_2d, is_same_quant );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SLICE_PARAM_NUM,
                reshape_tensors, input_num, &reshape_tensors[_INPUT_NUM], output_num );
            node_params[SCALAR_SAMLEFL_VALUE] = vsi_nn_kernel_scalar_create( graph, I32, &is_same_quant );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SLICE_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SAMLEFL_VALUE] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( slice, _setup )
