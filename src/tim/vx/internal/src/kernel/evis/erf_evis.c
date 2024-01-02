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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define HASH_UNARY_KEY(_input_type, _output_type, _image_2d) \
    ( (_input_type << 12) | (_output_type << 4) | (_image_2d))

#define KERNEL_SOURCE    "erf",

#define HASH_UNARY_SH_KERNEL_NAME( SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.erf_"#SRC_TYPE"to"#DST_TYPE)

#define TENSOR_UNARY_KERNELS(SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(SRC_TYPE, OUT_TYPE, 0), \
        HASH_UNARY_SH_KERNEL_NAME(SRC_TYPE, OUT_TYPE), \
        KERNEL_SOURCE },

#define HASH_UNARY_SH_KERNEL_2D_NAME(SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.erf_"#SRC_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_UNARY_KERNELS_2D(SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(SRC_TYPE, OUT_TYPE, 1), \
        HASH_UNARY_SH_KERNEL_2D_NAME(SRC_TYPE, OUT_TYPE), \
        KERNEL_SOURCE },


typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _erf_kernel_map[] =
{
    // Register kernel here
    TENSOR_UNARY_KERNELS(F16,  F16 )
    TENSOR_UNARY_KERNELS(F16,  I16 )
    TENSOR_UNARY_KERNELS(F16,  U8  )
    TENSOR_UNARY_KERNELS(F16,  I8  )
    TENSOR_UNARY_KERNELS(I16,  I16 )
    TENSOR_UNARY_KERNELS(I16,  F16 )
    TENSOR_UNARY_KERNELS(U8,   U8  )
    TENSOR_UNARY_KERNELS(U8,   F16 )
    TENSOR_UNARY_KERNELS(I8,   I8  )
    TENSOR_UNARY_KERNELS(I8,   F16 )
    TENSOR_UNARY_KERNELS(BF16, BF16)

    TENSOR_UNARY_KERNELS_2D(F16,  F16 )
    TENSOR_UNARY_KERNELS_2D(F16,  I16 )
    TENSOR_UNARY_KERNELS_2D(F16,  U8  )
    TENSOR_UNARY_KERNELS_2D(F16,  I8  )
    TENSOR_UNARY_KERNELS_2D(I16,  I16 )
    TENSOR_UNARY_KERNELS_2D(I16,  F16 )
    TENSOR_UNARY_KERNELS_2D(U8,   U8  )
    TENSOR_UNARY_KERNELS_2D(U8,   F16 )
    TENSOR_UNARY_KERNELS_2D(I8,   I8  )
    TENSOR_UNARY_KERNELS_2D(I8,   F16 )
    TENSOR_UNARY_KERNELS_2D(BF16, BF16)
};


/*
 * Kernel params
 */
static vx_param_description_t _erf_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _ERF_PARAM_NUM  _cnt_of_array( _erf_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_erf_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr[2]   = { NULL, NULL };
    vsi_size_array_t * out_shape             = NULL;
    float    inputScale                     = 1.0f;
    float    inputTail                      = 0;
    float    outputScale                    = 1.0f;
    float    outputZP                       = 0;
    uint32_t pack_key;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_shape  = attr[1]->shape;

    inputScale  = attr[0]->scale;
    inputTail   = 0 - (float)attr[0]->zero_point * inputScale;
    outputScale = (float)1.0f / attr[1]->scale;
    outputZP    = (float)attr[1]->zero_point;

#define _PACK_SELECT_KEY( IN_TYPE, OUT_TYPE )    \
        ( ( IN_TYPE << 16) | ( OUT_TYPE << 8))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype );

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    switch ( pack_key )
    {
        case _PACK_SELECT_KEY( BF16, BF16 ):
        {
            gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x01050004, 0x03070206, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtractOddData_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x07050301, 0x07050301, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
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
            gpu_dp_inst_t uniDatatoFp32Part0_4x4 = {{
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
                    "uniDatatoFp32Part0_4x4", &uniDatatoFp32Part0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "inputScale", &inputScale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "inputTail", &inputTail );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "outputScale", &outputScale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "outputZP", &outputZP );

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
        break;
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _erf_initializer() */


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
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _erf_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _erf_kernel_map );
    vx_param_description_t * param_def  = _erf_kernel_param_def;
    vx_kernel_initialize_f  initializer = _erf_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_UNARY_KEY( in_dtype, out_dtype, image_2d );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }

    if ( i < (uint32_t)kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = _cnt_of_array( _erf_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_ERF_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* rs_tensors[2] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t new_rank = 0;
    vsi_bool image_2d = FALSE;
    vsi_bool ret = FALSE;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(params);

    ret = vsi_nn_kernel_optimize_element_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );
    if ( ret )
    {
        rs_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shape, new_rank );
        rs_tensors[1] = vsi_nn_reshape_tensor( graph,
                outputs[0], shape, new_rank );
    }

    if ( !vsi_nn_kernel_gpu_check_shape( rs_tensors[0]->attr.size,
                rs_tensors[0]->attr.dim_num ) )
    {
        goto OnError;
    }

    image_2d = (rs_tensors[0]->attr.dim_num == 2 || rs_tensors[0]->attr.size[2] == 1);

    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ERF_PARAM_NUM,
                    rs_tensors, 1, &rs_tensors[1], 1 );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ERF_PARAM_NUM );
        }
    }

OnError:
    if (rs_tensors[0])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[0] );
    }

    if (rs_tensors[1])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[1] );
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( erf, _setup )
