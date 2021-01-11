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
#include "libnnext/vx_lib_nnext.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;

#define _A_GRUCELL_ACTIVATION_SMA_KERNEL_SOURCE      "grucell_activation_sma"

#define STR(a) #a

// Add kernel hashtable here
#define A_GRUCELL_ACTIVATION_SMA_HASH_KEY(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, _image_2d) \
        ((IN2_DTYPE << 24) | (IN1_DTYPE << 16) | ( IN0_DTYPE << 8 ) | ( OUT_DTYPE << 1) | (_image_2d))

#define A_GRUCELL_ACTIVATION_SMA_SH_KERNEL_NAME(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE) \
        CVIVANTE_NAMESPACE("evis.grucell_activation_sma_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE) \
            "_"STR(IN2_DTYPE)"to"STR(OUT_DTYPE))

#define PACK_KERNEL_MAP(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE) \
        { A_GRUCELL_ACTIVATION_SMA_HASH_KEY(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, IMAGE), \
        A_GRUCELL_ACTIVATION_SMA_SH_KERNEL_NAME(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE), \
        _A_GRUCELL_ACTIVATION_SMA_KERNEL_SOURCE}

#define A_GRUCELL_ACTIVATION_SMA_SH_KERNEL_2D_NAME(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE) \
        CVIVANTE_NAMESPACE("evis.grucell_activation_sma_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE) \
                        "_"STR(IN2_DTYPE)"to"STR(OUT_DTYPE)"_2D")

#define PACK_KERNEL_MAP_2D(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE) \
        { A_GRUCELL_ACTIVATION_SMA_HASH_KEY(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, IMAGE_2D), \
        A_GRUCELL_ACTIVATION_SMA_SH_KERNEL_2D_NAME(IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE), \
        _A_GRUCELL_ACTIVATION_SMA_KERNEL_SOURCE}

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _grucell_activation_sma_kernel_map[] =
{
    PACK_KERNEL_MAP(F16, F16,  F16,  F16),

    PACK_KERNEL_MAP_2D(F16, F16, F16, F16),
};

/*
 * Kernel params
 */
static vx_param_description_t _grucell_activation_sma_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _A_GRUCELL_ACTIVATION_SMA_PARAM_NUM  _cnt_of_array( _grucell_activation_sma_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_grucell_activation_sma_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
#define _PACK_A_GRUCELL_ACTIVATION_SMA_KEY( IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE )    \
        (( IN1_TYPE << 24) | ( IN1_TYPE << 16) | ( IN0_TYPE << 8) | ( OUT_TYPE))
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vx_tensor     input0                        = (vx_tensor)param[0];
    vx_tensor     input1                        = (vx_tensor)param[1];
    vx_tensor     input2                        = (vx_tensor)param[2];
    vx_tensor     output                        = (vx_tensor)param[3];
    uint32_t      i                             = 0;

    vsi_nn_kernel_tensor_attr_t * attr[4]       = { NULL, NULL, NULL, NULL };
    vsi_int_array_t             *output_shape   = NULL;
    uint32_t pack_key                           = 0;


    attr[0]  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input0);
    CHECK_PTR_FAIL_GOTO( attr[0], "vsi_nn_kernel_tensor_attr_create fail.", final );
    attr[1]  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input1);
    CHECK_PTR_FAIL_GOTO( attr[1], "vsi_nn_kernel_tensor_attr_create fail.", final );
    attr[2]  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input2);
    CHECK_PTR_FAIL_GOTO( attr[2], "vsi_nn_kernel_tensor_attr_create fail.", final );
    attr[3]  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output);
    CHECK_PTR_FAIL_GOTO( attr[3], "vsi_nn_kernel_tensor_attr_create fail.", final );


    pack_key = _PACK_A_GRUCELL_ACTIVATION_SMA_KEY( attr[0]->dtype, attr[1]->dtype, attr[2]->dtype, attr[3]->dtype );

    output_shape  = attr[3]->shape;

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1)
                                             / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = output_shape->data[1];
    gpu_param.global_size[2]   = output_shape->size > 2 ? output_shape->data[2] : 1;


    switch( pack_key )
    {
        case _PACK_A_GRUCELL_ACTIVATION_SMA_KEY( F16, F16, F16, F16 ):
        {
            gpu_dp_inst_t uniA_Times_B_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniA_Plus_B_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
                0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniA_Minus_B_2x8 = {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
                0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniA_Times_B_2x8", &uniA_Times_B_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniA_Plus_B_2x8", &uniA_Plus_B_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniA_Minus_B_2x8", &uniA_Minus_B_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        break;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    for ( i = 0; i < 4; i++)
    {
        if (attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
    }

#undef  _PACK_A_GRUCELL_ACTIVATION_SMA_KEY
    return status;
} /* _grucell_activation_sma_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _grucell_activation_sma_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _grucell_activation_sma_kernel_map );
    vx_param_description_t * param_def  = _grucell_activation_sma_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _grucell_activation_sma_kernel_param_def );
    vx_kernel_initialize_f  initializer = _grucell_activation_sma_initializer;
    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype   = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    in2_dtype   = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype   = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = A_GRUCELL_ACTIVATION_SMA_HASH_KEY(in0_dtype, in1_dtype, in2_dtype, out_dtype, image_2d);

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if( i < kernel_map_size )
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

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (2)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)

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
    vsi_nn_kernel_node_param_t node_params[_A_GRUCELL_ACTIVATION_SMA_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    int32_t* shapes_in[_INPUT_NUM];
    size_t rank_in[_INPUT_NUM];
    int32_t* shapes_ptr[_IO_NUM];
    int32_t  shapes[_IO_NUM][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    uint32_t new_rank = 0;
    int32_t  i        = 0;
    vsi_bool ret      = FALSE;
    vsi_nn_tensor_t* reshape_tensors[_IO_NUM] = { NULL };

    for (i = 0; i < _IO_NUM; i++)
    {
        shapes_ptr[i] = shapes[i];
    }

    for (i = 0; i < _INPUT_NUM; i++)
    {
        shapes_in[i] = (int32_t *)inputs[i]->attr.size;
        rank_in[i]   = (size_t)inputs[i]->attr.dim_num;
    }

    ret = vsi_nn_kernel_optimize_broadcast_shape(
            (const int32_t**)shapes_in, (const size_t*)rank_in, _INPUT_NUM,
            (int32_t *)outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes_ptr, shapes[_INPUT_NUM], &new_rank);

    if( ret )
    {
        for (i = 0; i < _INPUT_NUM; i++)
        {
            reshape_tensors[i] = vsi_nn_reshape_tensor( graph,
                    inputs[i], (uint32_t*)shapes[i], new_rank );
        }

        for (i = 0; i < _OUTPUT_NUM; i++)
        {
            reshape_tensors[i + _INPUT_NUM] = vsi_nn_reshape_tensor( graph,
                    outputs[i], (uint32_t*)shapes[_INPUT_NUM], new_rank );
        }
    }
    else
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)reshape_tensors[_INPUT_NUM]->attr.size,
                reshape_tensors[_INPUT_NUM]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (reshape_tensors[_INPUT_NUM]->attr.dim_num == 2 || reshape_tensors[_INPUT_NUM]->attr.size[2] == 1);
    status = _query_kernel( kernel, reshape_tensors, &reshape_tensors[_INPUT_NUM], image_2d);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _A_GRUCELL_ACTIVATION_SMA_PARAM_NUM,
                    reshape_tensors, input_num, &reshape_tensors[_INPUT_NUM], output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _A_GRUCELL_ACTIVATION_SMA_PARAM_NUM );
        }
    }

    for (i = 0; i < _IO_NUM; i++)
    {
        if (reshape_tensors[i])
        {
            vsi_nn_ReleaseTensor( &reshape_tensors[i] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( grucell_activation_sma, _setup )
