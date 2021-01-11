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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/** Unary Kernel internal type */
typedef enum
{
    UNARY_SIN,
    UNARY_EXP,
    UNARY_LOG,
    UNARY_ELU,
    UNARY_NEG,
    UNARY_HSIGMOID,
    UNARY_MISH,
} unary_type_e;

/*
 * Define kernel meta.
 */
#define HASH_UNARY_KEY(_type, _input_type, _output_type, _image_2d) \
    ((_type << 20) | (_input_type << 12) | (_output_type << 4) | (_image_2d))

 #define VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() \
    "eltwise_unary"

#define HASH_UNARY_SH_KERNEL_NAME(FUNC_NAME, SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl."#FUNC_NAME"_"#SRC_TYPE"to"#DST_TYPE)

#define TENSOR_UNARY_KERNELS(FUNC_NAME, TYPE, SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(TYPE, SRC_TYPE, OUT_TYPE, 0), \
        HASH_UNARY_SH_KERNEL_NAME(FUNC_NAME, SRC_TYPE, OUT_TYPE), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define HASH_UNARY_SH_KERNEL_2D_NAME(FUNC_NAME, SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl."#FUNC_NAME"_"#SRC_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_UNARY_KERNELS_2D(FUNC_NAME, TYPE, SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(TYPE, SRC_TYPE, OUT_TYPE, 1), \
        HASH_UNARY_SH_KERNEL_2D_NAME(FUNC_NAME, SRC_TYPE, OUT_TYPE), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define TENSOR_UNARY_KERNELS_FLOAT(FUNC_NAME, TYPE, SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(TYPE, SRC_TYPE, OUT_TYPE, 0), \
        HASH_UNARY_SH_KERNEL_NAME(FUNC_NAME, F32, F32), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define TENSOR_UNARY_KERNELS_FLOAT_2D(FUNC_NAME, TYPE, SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(TYPE, SRC_TYPE, OUT_TYPE, 1), \
        HASH_UNARY_SH_KERNEL_2D_NAME(FUNC_NAME, F32, F32), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define SIN_OPERATION           sin
#define EXP_OPERATION           exp
#define LOG_OPERATION           log
#define ELU_OPERATION           elu
#define NEG_OPERATION           neg
#define HSIGMOID_OPERATION      hard_sigmoid
#define MISH_OPERATION      mish

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_UNARY_KERNELS_FLOAT(SIN_OPERATION,      UNARY_SIN,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(SIN_OPERATION,      UNARY_SIN,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT(EXP_OPERATION,      UNARY_EXP,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(EXP_OPERATION,      UNARY_EXP,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT(LOG_OPERATION,      UNARY_LOG,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(LOG_OPERATION,      UNARY_LOG,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT(ELU_OPERATION,      UNARY_ELU,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(ELU_OPERATION,      UNARY_ELU,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT(NEG_OPERATION,      UNARY_NEG,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(NEG_OPERATION,      UNARY_NEG,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT(HSIGMOID_OPERATION, UNARY_HSIGMOID, F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT(MISH_OPERATION,     UNARY_MISH,     F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(MISH_OPERATION,     UNARY_MISH,     F16, F16)

    TENSOR_UNARY_KERNELS_FLOAT_2D(SIN_OPERATION,      UNARY_SIN,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(SIN_OPERATION,      UNARY_SIN,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT_2D(EXP_OPERATION,      UNARY_EXP,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(EXP_OPERATION,      UNARY_EXP,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT_2D(LOG_OPERATION,      UNARY_LOG,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(LOG_OPERATION,      UNARY_LOG,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT_2D(ELU_OPERATION,      UNARY_ELU,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(ELU_OPERATION,      UNARY_ELU,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT_2D(NEG_OPERATION,      UNARY_NEG,      F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(NEG_OPERATION,      UNARY_NEG,      F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16, F16)
    TENSOR_UNARY_KERNELS_FLOAT_2D(MISH_OPERATION,     UNARY_MISH,     F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(MISH_OPERATION,     UNARY_MISH,     F16, F16)

    TENSOR_UNARY_KERNELS(SIN_OPERATION,      UNARY_SIN,      U8,  U8)
    TENSOR_UNARY_KERNELS(EXP_OPERATION,      UNARY_EXP,      U8,  U8)
    TENSOR_UNARY_KERNELS(LOG_OPERATION,      UNARY_LOG,      U8,  U8)
    TENSOR_UNARY_KERNELS(ELU_OPERATION,      UNARY_ELU,      U8,  U8)
    TENSOR_UNARY_KERNELS(NEG_OPERATION,      UNARY_NEG,      U8,  U8)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, U8,  U8)
    TENSOR_UNARY_KERNELS(MISH_OPERATION,     UNARY_MISH,     U8,  U8)

    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION,      UNARY_SIN,      U8,  U8)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION,      UNARY_EXP,      U8,  U8)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION,      UNARY_LOG,      U8,  U8)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION,      UNARY_ELU,      U8,  U8)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION,      UNARY_NEG,      U8,  U8)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, U8,  U8)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION,     UNARY_MISH,     U8,  U8)

    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, I32,  I32)

    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, I32,  I32)
};

#undef SIN_OPERATION
#undef EXP_OPERATION
#undef LOG_OPERATION
#undef ELU_OPERATION
#undef NEG_OPERATION
#undef HSIGMOID_OPERATION
#undef MISH_OPERATION
/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_INPUT_SCALE           (2)
#define SCALAR_INPUT_TAIL            (3)
#define SCALAR_OUTPUT_SCALE          (4)
#define SCALAR_OUTPUT_ZP             (5)
#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_eltwise_unary_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    gpu_param_t gpu_param = {
        3,         // workdim
        {0, 0, 0}, // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0}, // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0}, // localWorkSize: local group size in thread
        {0, 0, 0}  // globalWorkSize: image size in thread
        };

    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_int_array_t * out_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_shape  = attr[1]->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _eltwise_unary_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t type,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_UNARY_KEY( type, input_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _eltwise_unary_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                kernel_map[i].source_name );
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
    vsi_nn_kernel_t             * kernel,
    const unary_type_e            unary_type
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* rs_tensors[2] = { NULL };
    int32_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    int32_t new_rank = 0;
    vsi_bool ret;

    float inputScale = inputs[0]->attr.dtype.scale;
    float inputTail = (float)inputs[0]->attr.dtype.zero_point * inputScale;
    float outputScale = outputs[0]->attr.dtype.scale;
    float outputZP = (float)outputs[0]->attr.dtype.zero_point + 0.5f;

    ret = vsi_nn_kernel_optimize_element_shape(
            (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );
    if( ret )
    {
        rs_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], (uint32_t*)shape, new_rank );
        rs_tensors[1] = vsi_nn_reshape_tensor( graph,
                outputs[0], (uint32_t*)shape, new_rank );
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)rs_tensors[0]->attr.size,
                rs_tensors[0]->attr.dim_num ) )
    {
        return NULL;
    }

    outputScale = vsi_abs(outputScale) < 1e-5 ? 0.0f : 1.0f / outputScale;

    image_2d = (rs_tensors[0]->attr.dim_num == 2 || rs_tensors[0]->attr.size[2] == 1);
    status = _query_kernel( rs_tensors, &rs_tensors[1], unary_type, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if( node )
        {
            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    rs_tensors, 1, &rs_tensors[1], 1 );
            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputScale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputTail );
            node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputScale );
            node_params[SCALAR_OUTPUT_ZP] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputZP );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, OnError );
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

    if (node_params[SCALAR_INPUT_SCALE])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
    }

    if (node_params[SCALAR_INPUT_TAIL])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
    }

    if (node_params[SCALAR_OUTPUT_SCALE])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
    }

    if (node_params[SCALAR_OUTPUT_ZP])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
    }

    return node;
} /* _setup() */

#define REGISTER_ELTWISE_UNARY_BACKEND_CL(KERNEL_NAME, UNARY_TYPE) \
    static vsi_nn_kernel_node_t _##KERNEL_NAME##_setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num, \
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ) \
    { \
        return _setup(graph, inputs, input_num, outputs, output_num, \
                params, kernel, UNARY_TYPE); \
    } \
    REGISTER_BACKEND_CL( KERNEL_NAME, _##KERNEL_NAME##_setup )


REGISTER_ELTWISE_UNARY_BACKEND_CL( sin,          UNARY_SIN )
REGISTER_ELTWISE_UNARY_BACKEND_CL( exp,          UNARY_EXP )
REGISTER_ELTWISE_UNARY_BACKEND_CL( log,          UNARY_LOG )
REGISTER_ELTWISE_UNARY_BACKEND_CL( elu,          UNARY_ELU )
REGISTER_ELTWISE_UNARY_BACKEND_CL( neg,          UNARY_NEG )
REGISTER_ELTWISE_UNARY_BACKEND_CL( hard_sigmoid, UNARY_HSIGMOID )
REGISTER_ELTWISE_UNARY_BACKEND_CL( mish,         UNARY_MISH )

__END_DECLS
