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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_GRUCELL_ACTIVATION,
} _internal_kernel_e;

#define _GRUCELL_ACTIVATION_KERNEL_SOURCE               "grucell_activation"
#define _GRUCELL_ACTIVATION_KERNEL_NAME                 CVIVANTE_NAMESPACE("evis.grucell_activation")

#define _CDNN_KERNEL_SOURCE0          "grucell_cdnn_activation"
#define _CDNN_KERNEL_SOURCE1          "grucell_cdnn_activation_u8"
#define _GRUCELL_ACTIVATION_CDNN_KERNEL_NAME            CVIVANTE_NAMESPACE("evis.grucell_cdnn_activation")

typedef enum _batch_fisrt_layerout_e
{
    NC,
    CN,
    CN_FULL,
} batch_fisrt_layerout_e;

typedef enum _gru_activation_type_e
{
    sigmoid = VSI_NN_ACT_SIGMOID,
    hsigmoid = VSI_NN_ACT_HARD_SIGMOID,
} gru_activation_type_e;

#define STR(a) #a
// Add kernel hashtable here
#define GRUCELL_ACTIVATION_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT ) \
        ((uint64_t)IN0_DTYPE | ( (uint64_t)IN1_DTYPE << 8 ) | ( (uint64_t)IN2_DTYPE << 16 ) \
        | ( (uint64_t)OUT_DTYPE << 24 ) | ( (uint64_t)GATE_ACT << 32 ) \
        | ( (uint64_t)CAND_ACT << 40 ) | ( (uint64_t)LAYER_OUT << 48 ))

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT ) \
{ GRUCELL_ACTIVATION_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT ), \
  CVIVANTE_NAMESPACE("evis.grucell_activation_"#IN0_DTYPE"_"#IN1_DTYPE"_"#IN2_DTYPE"_to_"#OUT_DTYPE"_"#GATE_ACT), \
  _GRUCELL_ACTIVATION_KERNEL_SOURCE }

#define PACK_KERNEL_CDNN_SEP_MAP( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT, SOURCE ) \
 { GRUCELL_ACTIVATION_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT ), \
    CVIVANTE_NAMESPACE("evis.grucell_activation_cdnn_sep_"#IN0_DTYPE"_"#IN1_DTYPE"_" \
        #IN2_DTYPE"_to_"#OUT_DTYPE"_"STR(LAYER_OUT)), \
   SOURCE }

#define PACK_KERNEL_CDNN_MAP( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT, SOURCE ) \
 { GRUCELL_ACTIVATION_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, GATE_ACT, CAND_ACT, LAYER_OUT ), \
   CVIVANTE_NAMESPACE("evis.grucell_activation_cdnn_"#IN0_DTYPE"_"#IN1_DTYPE"_"#IN2_DTYPE"_to_"#OUT_DTYPE), \
   SOURCE }

typedef struct
{
    uint64_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _grucell_activation_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8, U8, U8,     sigmoid, VSI_NN_ACT_TANH, NC),
    PACK_KERNEL_MAP( F16, F16, F16, F16, sigmoid, VSI_NN_ACT_TANH, NC),
    PACK_KERNEL_MAP( F16, F16, F16, U8,  sigmoid, VSI_NN_ACT_TANH, NC),

    PACK_KERNEL_MAP( U8, U8, U8, U8,     sigmoid, VSI_NN_ACT_TANH, CN),
    PACK_KERNEL_MAP( F16, F16, F16, F16, sigmoid, VSI_NN_ACT_TANH, CN),
    PACK_KERNEL_MAP( F16, F16, F16, U8,  sigmoid, VSI_NN_ACT_TANH, CN),

    PACK_KERNEL_MAP( U8, U8, U8, U8,     hsigmoid, VSI_NN_ACT_TANH, NC),
    PACK_KERNEL_MAP( F16, F16, F16, F16, hsigmoid, VSI_NN_ACT_TANH, NC),
    PACK_KERNEL_MAP( F16, F16, F16, U8,  hsigmoid, VSI_NN_ACT_TANH, NC),

    PACK_KERNEL_MAP( U8, U8, U8, U8,     hsigmoid, VSI_NN_ACT_TANH, CN),
    PACK_KERNEL_MAP( F16, F16, F16, F16, hsigmoid, VSI_NN_ACT_TANH, CN),
    PACK_KERNEL_MAP( F16, F16, F16, U8,  hsigmoid, VSI_NN_ACT_TANH, CN),
};

static const _kernel_map_type _grucell_cunn_sep_activation_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_CDNN_SEP_MAP( F16, F16, F16, F16, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, NC, _CDNN_KERNEL_SOURCE0 ),

    PACK_KERNEL_CDNN_SEP_MAP( F16, F16, F16, F16, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, CN, _CDNN_KERNEL_SOURCE0 ),

    PACK_KERNEL_CDNN_SEP_MAP( F16, F16, F16, F16, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, CN_FULL, _CDNN_KERNEL_SOURCE0 ),

    PACK_KERNEL_CDNN_SEP_MAP( U8, U8, U8, U8, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, NC, _CDNN_KERNEL_SOURCE1 ),

    PACK_KERNEL_CDNN_SEP_MAP( U8, U8, U8, U8, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, CN, _CDNN_KERNEL_SOURCE1 ),

    PACK_KERNEL_CDNN_SEP_MAP( U8, U8, U8, U8, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, CN_FULL, _CDNN_KERNEL_SOURCE1 ),
};

static const _kernel_map_type _grucell_cunn_activation_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_CDNN_MAP( F16, F16, F16, F16, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, NC, _CDNN_KERNEL_SOURCE0 ),

    PACK_KERNEL_CDNN_MAP( F16, F16, F16, F16, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, CN, _CDNN_KERNEL_SOURCE0 ),

    PACK_KERNEL_CDNN_MAP( U8, U8, U8, U8, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, NC, _CDNN_KERNEL_SOURCE1 ),

    PACK_KERNEL_CDNN_MAP( U8, U8, U8, U8, VSI_NN_ACT_SIGMOID, VSI_NN_ACT_TANH, CN, _CDNN_KERNEL_SOURCE1 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _grucell_activation_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GRUCELL_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_kernel_param_def )

static vx_param_description_t _grucell_activation_separated_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GRUCELL_CDNN_SEP_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_separated_kernel_param_def )

static vx_param_description_t _grucell_activation_cdnn_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GRUCELL_CDNN_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_cdnn_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_grucell_activation_initializer)
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
    float    tensorScale[4]                 = {1.0f, 1.0f, 1.0f, 1.0f};
    float    tensorZP[4]                    = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t  i                             = 0;
    uint32_t  pack_key                      = 0;
    vsi_int_array_t * output_shape          = NULL;
    vsi_nn_kernel_tensor_attr_t * attr[4]   = { NULL, NULL, NULL, NULL };

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
    attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );

    for (i = 0; i < 4; i++)
    {
        if( attr[i]->quant == VSI_NN_KERNEL_QUANT_ASYMM
            || attr[i]->quant == VSI_NN_KERNEL_QUANT_SYMM)
        {
            tensorZP[i]     = (float)attr[i]->asymm.zero_point;
            tensorScale[i]  = attr[i]->asymm.scale;
        }
    }

    tensorZP[0] = tensorScale[0] * tensorZP[0];
    tensorZP[1] = tensorScale[1] * tensorZP[1];
    tensorZP[2] = tensorScale[2] * tensorZP[2];
    tensorScale[3] = 1.0f / tensorScale[3];

    output_shape  = attr[3]->shape;

    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0]  =
        gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]  = output_shape->data[1];

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 8) | (IN2_TYPE << 16) | ( OUT_TYPE << 24))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype,
                attr[2]->dtype, attr[3]->dtype );

    switch (pack_key)
    {
        case _PACK_SELECT_KEY( F16, F16, F16, F16 ):
        case _PACK_SELECT_KEY( F16, F16, F16, U8 ):
        case _PACK_SELECT_KEY( U8, U8, U8, U8 ):
            {
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
                gpu_dp_inst_t uniConvDatatoFp32_4x4 = {{
                    0x01010101, // TCfg
                    0x00000000, // ASelt
                    0x00010000, 0x00030002, // ABin
                    0x02020202, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
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

                if (attr[3]->dtype == F16)
                {
                    status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtract8Data_2x8", &uniExtractHalf8_2x8 );
                }
                else
                {
                    status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
                }
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvDatatoFp32_4x4", &uniConvDatatoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "tensorZP", &tensorZP );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "tensorScale", &tensorScale );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
    default:
        break;
    }

    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    for (i = 0; i < 4; i++)
    {
        if (attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
    }

    return status;
} /* _grucell_activation_initializer() */

DEF_KERNEL_INITIALIZER(_grucell_activation_cdnn_initializer)
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
    uint32_t  i                             = 0;
    uint32_t  pack_key                      = 0;
    int32_t   layer_out                     = 1;
    int32_t   input_size                    = 1;
    int32_t   batch                         = 1;
    float     input_scale                   = 1.0f;
    float     input_tail                    = 0;
    float     output_scale                  = 1.0f;
    float     output_zp                     = 0;
    float     input_r_scale                 = 1.0f;
    float     input_r_tail                  = 0;
    float     recur_r_scale                 = 1.0f;
    float     recur_r_tail                  = 0;
    float     input_z_scale                 = 1.0f;
    float     input_z_tail                  = 0;
    float     recur_z_scale                 = 1.0f;
    float     recur_z_tail                  = 0;
    float     input_c_scale                 = 1.0f;
    float     input_c_tail                  = 0;
    float     recur_c_scale                 = 1.0f;
    float     recur_c_tail                  = 0;
    vsi_int_array_t * output_shape          = NULL;
    vsi_nn_kernel_tensor_attr_t * attr[8]   = { NULL };

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    if ( param_size == _GRUCELL_CDNN_SEP_ACTIVATION_PARAM_NUM )
    {
        attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[4] );
        CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
        attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[param_size - 4] );
        CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );

        attr[4] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
        CHECK_PTR_FAIL_GOTO( attr[4], "Create tensor attr buffer fail.", final );
        attr[5] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[5] );
        CHECK_PTR_FAIL_GOTO( attr[5], "Create tensor attr buffer fail.", final );
        attr[6] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
        CHECK_PTR_FAIL_GOTO( attr[6], "Create tensor attr buffer fail.", final );
        attr[7] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[6] );
        CHECK_PTR_FAIL_GOTO( attr[7], "Create tensor attr buffer fail.", final );
    }
    else
    {
        attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
        CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
        attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[param_size - 4] );
        CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );
    }

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[param_size - 1], &layer_out);
    CHECK_STATUS_FAIL_GOTO(status, final );

    output_shape  = attr[3]->shape;

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM
     || attr[0]->quant == VSI_NN_KERNEL_QUANT_SYMM )
    {
        input_scale  = attr[0]->asymm.scale;
        input_tail   = 0 - input_scale * (float)attr[0]->asymm.zero_point;
    }

    if( attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM
     || attr[1]->quant == VSI_NN_KERNEL_QUANT_SYMM )
    {
        input_r_scale  = attr[1]->asymm.scale;
        input_r_tail   = 0 - input_r_scale * (float)attr[1]->asymm.zero_point;
    }

    if( attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM
     || attr[2]->quant == VSI_NN_KERNEL_QUANT_SYMM )
    {
        recur_r_scale  = attr[2]->asymm.scale;
        recur_r_tail   = 0 - recur_r_scale * (float)attr[2]->asymm.zero_point;
    }

    if( attr[3]->quant == VSI_NN_KERNEL_QUANT_ASYMM
     || attr[3]->quant == VSI_NN_KERNEL_QUANT_SYMM )
    {
        output_scale  = 1.0f / attr[3]->asymm.scale;
        output_zp   = (float)attr[3]->asymm.zero_point;
    }

    if ( param_size == _GRUCELL_CDNN_SEP_ACTIVATION_PARAM_NUM )
    {
        if( attr[4]->quant == VSI_NN_KERNEL_QUANT_ASYMM
         || attr[4]->quant == VSI_NN_KERNEL_QUANT_SYMM )
        {
            input_z_scale  = attr[4]->asymm.scale;
            input_z_tail   = 0 - input_z_scale * (float)attr[4]->asymm.zero_point;
        }

        if( attr[5]->quant == VSI_NN_KERNEL_QUANT_ASYMM
         || attr[5]->quant == VSI_NN_KERNEL_QUANT_SYMM )
        {
            recur_z_scale  = attr[5]->asymm.scale;
            recur_z_tail   = 0 - recur_z_scale * (float)attr[5]->asymm.zero_point;
        }

        if( attr[6]->quant == VSI_NN_KERNEL_QUANT_ASYMM
         || attr[6]->quant == VSI_NN_KERNEL_QUANT_SYMM )
        {
            input_c_scale  = attr[6]->asymm.scale;
            input_c_tail   = 0 - input_c_scale * (float)attr[6]->asymm.zero_point;
        }

        if( attr[5]->quant == VSI_NN_KERNEL_QUANT_ASYMM
         || attr[5]->quant == VSI_NN_KERNEL_QUANT_SYMM )
        {
            recur_c_scale  = attr[7]->asymm.scale;
            recur_c_tail   = 0 - recur_c_scale * (float)attr[7]->asymm.zero_point;
        }
    }

    if (layer_out == 1 || layer_out == 2)
    {
        input_size = attr[1]->shape->data[0];
        batch      = attr[1]->shape->data[1];
    }
    else
    {
        input_size = output_shape->data[0];
        batch      = output_shape->data[1];
    }

    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0]  = (input_size + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0];
    gpu_param.global_size[1]  = batch;

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 8) | (IN2_TYPE << 16) | ( OUT_TYPE << 24))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype,
                attr[2]->dtype, attr[3]->dtype );

    switch (pack_key)
    {
        case _PACK_SELECT_KEY( F16, F16, F16, F16 ):
            {
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
                gpu_dp_inst_t uniConvDatatoFp32_4x4 = {{
                    0x01010101, // TCfg
                    0x00000000, // ASelt
                    0x00010000, 0x00030002, // ABin
                    0x02020202, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
                }, GPU_DP_TYPE_16 };
                gpu_dp_inst_t uiF16AddF16_4x4 = {{
                    0x05050505, // TCfg
                    0x04040404, // ASelt
                    0x00110000, 0x00330022, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                    0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
                }, GPU_DP_TYPE_16 };


                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniExtract8Data_2x8", &uniExtractHalf8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvDatatoFp32_4x4", &uniConvDatatoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uiF16AddF16_4x4", &uiF16AddF16_4x4 );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        case _PACK_SELECT_KEY( U8, U8, U8, U8 ):
            {
                gpu_dp_inst_t uniExtractInteger_2x8 = {{
                    0x33333333, // TCfg
                    0x11110000, // ASelt
                    0x03020100, 0x03020100, // ABin
                    0x00000000, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00002400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                }, GPU_DP_TYPE_16};
                gpu_dp_inst_t uniConvDatatoFp32_4x4 = {{
                    0x01010101, // TCfg
                    0x00000000, // ASelt
                    0x00010000, 0x00030002, // ABin
                    0x02020202, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                    0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
                }, GPU_DP_TYPE_16 };

                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvDatatoFp32_4x4", &uniConvDatatoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_scale", &input_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_tail", &input_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_r_scale", &input_r_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_r_tail", &input_r_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "recur_r_scale", &recur_r_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "recur_r_tail", &recur_r_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_z_scale", &input_z_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_z_tail", &input_z_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "recur_z_scale", &recur_z_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "recur_z_tail", &recur_z_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_z_scale", &input_z_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_c_scale", &input_c_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "input_c_tail", &input_c_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "recur_c_scale", &recur_c_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "recur_c_tail", &recur_c_tail );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "output_scale", &output_scale );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "output_zp", &output_zp );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
    default:
        break;
    }

    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    for (i = 0; i < 8; i++)
    {
        if (attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
    }

    return status;
} /* _grucell_activation_cdnn_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t gate_activation,
    int32_t candidate_activation,
    int32_t input_category,
    int32_t input_layout,
    int32_t use_cudnn,
    int32_t* param_count,
    int32_t* input_count,
    int32_t* output_count
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    batch_fisrt_layerout_e layer_out;
    _kernel_map_type * kernel_map = (_kernel_map_type *)_grucell_activation_kernel_map;
    vx_param_description_t * param_def  = _grucell_activation_kernel_param_def;
    vx_kernel_initialize_f  initializer = _grucell_activation_initializer;
    int32_t numParams = 0;
    int32_t kernel_num = 0;
    uint64_t key = 0;
    int32_t  i = 0;

    layer_out = input_layout == GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC ? NC :
                input_layout == GRUCELL_ACTIVATION_INPUT_LAYOUT_INPUT_NC_FC_CN ? CN : CN_FULL;

    if (use_cudnn)
    {
        in0_dtype  = vsi_nn_kernel_map_dtype( inputs[GRUCELL_ACTIVATION_INPUT_H_STATE]->attr.dtype.vx_type );
        in1_dtype  = vsi_nn_kernel_map_dtype( inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr.dtype.vx_type );
        in2_dtype  = vsi_nn_kernel_map_dtype( inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R]->attr.dtype.vx_type );
        out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

        if (input_category == GRUCELL_INPUT_CATEGORY_CUDNN)
        {
            *param_count = _GRUCELL_CDNN_SEP_ACTIVATION_PARAM_NUM;
            *input_count = 13;
            *output_count = 2;

            numParams  = _GRUCELL_CDNN_SEP_ACTIVATION_PARAM_NUM;
            kernel_num = _cnt_of_array(_grucell_cunn_sep_activation_kernel_map);
            kernel_map = (_kernel_map_type *)_grucell_cunn_sep_activation_kernel_map;
            param_def  = _grucell_activation_separated_kernel_param_def;
            initializer  = _grucell_activation_cdnn_initializer;
        }
        else
        {
            *param_count = _GRUCELL_CDNN_ACTIVATION_PARAM_NUM;
            *input_count = 9;
            *output_count = 2;

            numParams  = _GRUCELL_CDNN_ACTIVATION_PARAM_NUM;
            kernel_num = _cnt_of_array(_grucell_cunn_activation_kernel_map);
            kernel_map = (_kernel_map_type *)_grucell_cunn_activation_kernel_map;
            param_def  = _grucell_activation_cdnn_kernel_param_def;
            initializer  = _grucell_activation_cdnn_initializer;
        }
    }
    else
    {
        *param_count = _GRUCELL_ACTIVATION_PARAM_NUM;
        *input_count = 3;
        *output_count = 2;

        numParams  = _GRUCELL_ACTIVATION_PARAM_NUM;
        kernel_num = _cnt_of_array(_grucell_activation_kernel_map);
        in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
        in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
        in2_dtype  = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
        out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    }

    key = GRUCELL_ACTIVATION_HASH_KEY( in0_dtype, in1_dtype, in2_dtype, out_dtype,
        gate_activation, candidate_activation, layer_out );

    for( i = 0; i < kernel_num; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < kernel_num )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)numParams;
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
    vsi_nn_kernel_node_param_t* node_params = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t** _inputs = NULL;
    vsi_nn_tensor_t* _biases[6] = {NULL};
    vsi_nn_tensor_t* _fc_r[2] = {NULL};
    vsi_nn_tensor_t* cond_zeros = NULL;
    int32_t i = 0;
    int32_t j = 0;
    int32_t k = 0;
    int32_t input_size = inputs[0]->attr.size[0];
    int32_t batch = inputs[0]->attr.size[1];
    int32_t param_count = 0;
    int32_t input_count = 0;
    int32_t output_count = 0;
    int32_t gate_activation = 0;
    int32_t candidate_activation = 0;
    int32_t input_category = vsi_nn_kernel_param_get_int32( params, "input_category" );
    int32_t use_cudnn = vsi_nn_kernel_param_get_int32( params, "use_cudnn_implementation" );
    int32_t input_layout = vsi_nn_kernel_param_get_int32( params, "input_layout" );

    gate_activation = vsi_nn_kernel_param_get_int32( params, "gate_activation" );
    candidate_activation = vsi_nn_kernel_param_get_int32( params, "candidate_activation" );

    if (use_cudnn && inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z] == NULL)
    {
        input_category = GRUCELL_INPUT_CATEGORY_DEFAULT;
    }
    else if (use_cudnn && inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z])
    {
        input_category = GRUCELL_INPUT_CATEGORY_CUDNN;
    }

    status = _query_kernel( kernel, inputs, outputs, gate_activation, candidate_activation,
        input_category, input_layout, use_cudnn, &param_count, &input_count, &output_count );

    if( VSI_SUCCESS == status)
    {
        _inputs = (vsi_nn_tensor_t**)malloc(input_count * sizeof(vsi_nn_tensor_t**));
        node_params = (vsi_nn_kernel_node_param_t *)malloc(sizeof(vsi_nn_kernel_node_param_t) * param_count);

        if (use_cudnn)
        {
            vsi_nn_tensor_attr_t attr;

            if (input_category == GRUCELL_INPUT_CATEGORY_DEFAULT)
            {
                memcpy(&attr, &inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R]->attr, sizeof(vsi_nn_tensor_attr_t));
                attr.size[0] = input_size;
                attr.size[1] = 3 * batch;
                attr.dim_num = 2;

                _fc_r[0] = vsi_nn_reshape_tensor(graph,
                    inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R], attr.size, attr.dim_num);
                inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R] = _fc_r[0];
                _fc_r[1] = vsi_nn_reshape_tensor(graph,
                    inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R], attr.size, attr.dim_num);
                inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R] = _fc_r[1];
            }

            memcpy(&attr, &inputs[GRUCELL_ACTIVATION_INPUT_BIAS_R]->attr, sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            for ( i = 0; i < 3; i++)
            {
                _biases[i] = vsi_nn_reshape_tensor(graph,
                    inputs[GRUCELL_ACTIVATION_INPUT_BIAS_R + i], attr.size, attr.dim_num);
                inputs[GRUCELL_ACTIVATION_INPUT_BIAS_R + i] = _biases[i];
            }

            if(!inputs[GRUCELL_ACTIVATION_INPUT_COND_R] || !inputs[GRUCELL_ACTIVATION_INPUT_COND_Z]
                || !inputs[GRUCELL_ACTIVATION_INPUT_COND_C])
            {
                cond_zeros = vsi_nn_CreateTensorWithDefault(graph, &attr, 0.0);
            }
            for ( i = 0; i < 3; i++)
            {
                if (inputs[GRUCELL_ACTIVATION_INPUT_COND_R + i])
                {
                    /* Shader kernel cannot take 1-d tensors as inptus
                      reshape them to 2-d to workaround */
                    if(1 == inputs[GRUCELL_ACTIVATION_INPUT_COND_R + i]->attr.dim_num)
                    {
                        _biases[i + 3] = vsi_nn_reshape_tensor(graph,
                        inputs[GRUCELL_ACTIVATION_INPUT_COND_R + i], attr.size, attr.dim_num);
                        inputs[GRUCELL_ACTIVATION_INPUT_COND_R + i] = _biases[i + 3];
                    }
                    else
                    {
                        /* high level had done the workaround */
                    }
                }
                else
                {
                    inputs[GRUCELL_ACTIVATION_INPUT_COND_R + i] = cond_zeros;
                }
            }
        }

        for(i = 0, k = 0; k < input_count; i++)
        {
            if (inputs[i])
            {
                _inputs[k ++] = inputs[i];
            }
        }

        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            j = input_count + output_count;

            vsi_nn_kernel_node_pack_io( node_params, param_count,
                    _inputs, input_count, outputs, output_num );
            /* Pass parameters to node. */
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &gate_activation );
            node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &candidate_activation );
            if( use_cudnn )
            {
                node_params[j++] = vsi_nn_kernel_scalar_create(graph, I32, &input_layout );
            }
            status  = vsi_nn_kernel_node_pass_param( node, node_params, param_count );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            vsi_nn_kernel_scalar_release( &node_params[--j] );
            if( use_cudnn )
            {
                vsi_nn_kernel_scalar_release( &node_params[--j] );
            }
        }
        if(cond_zeros)
        {
            vsi_nn_ReleaseTensor(&cond_zeros);
        }

        for ( i = 0; i < 6; i++)
        {
            if (_biases[i])
            {
                vsi_nn_ReleaseTensor(&_biases[i]);
            }
        }

        for ( i = 0; i < 2; i++)
        {
            if (_fc_r[i])
            {
                vsi_nn_ReleaseTensor(&_fc_r[i]);
            }
        }
    }

    vsi_nn_safe_free(_inputs);
    vsi_nn_safe_free(node_params);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( grucell_activation, _setup )


