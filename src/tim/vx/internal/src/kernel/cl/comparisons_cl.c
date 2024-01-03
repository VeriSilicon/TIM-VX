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

#if !(VX_RELATIONAL_OPS_VX_SUPPORT_EXT)
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

/** Comparisons Kernel internal type */
typedef enum
{
    COMP_GREAT = VSI_NN_RELATIONAL_OPS_GREAT,
    COMP_GREAT_EQUAL = VSI_NN_RELATIONAL_OPS_GREAT_EQUAL,
    COMP_LESS = VSI_NN_RELATIONAL_OPS_LESS,
    COMP_LESS_EQUAL = VSI_NN_RELATIONAL_OPS_LESS_EQUAL,
    COMP_NOT_EQUAL = VSI_NN_RELATIONAL_OPS_NOT_EQUAL,
    COMP_EQUAL = VSI_NN_RELATIONAL_OPS_EQUAL,
} relational_type_e;

/*
 * Define kernel meta.
 */
#define HASH_COMPARISONS_KEY(_type, _input0_type, _input1_type, _output_type, _image_2d) \
    ((_type << 28) | (_input0_type << 20) | (_input1_type << 12)  | (_output_type << 2) | (_image_2d))

#define KERNEL_SOURCE_2D    "relational_ops",
#define KERNEL_SOURCE_3D    "relational_ops",

#define HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, SRC0_TYPE, SRC1_TYPE) \
    CVIVANTE_NAMESPACE("cl."#FUNC_NAME"_"#SRC0_TYPE#SRC1_TYPE"toBOOL8")

#define COMPARISONS_KERNELS_U32(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 0), \
        HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, U32, U32), \
        SOURCE },

#define HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, SRC0_TYPE, SRC1_TYPE) \
    CVIVANTE_NAMESPACE("cl."#FUNC_NAME"_"#SRC0_TYPE#SRC1_TYPE"toBOOL8_2D")

#define COMPARISONS_KERNELS_U32_2D(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 1), \
        HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, U32, U32), \
        SOURCE },

#define COMPARISONS_KERNELS_I32(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 0), \
    HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, I32, I32), \
        SOURCE },

#define COMPARISONS_KERNELS_I32_2D(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 1), \
        HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, I32, I32), \
        SOURCE },

#define COMPARISONS_KERNELS_F32(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 0), \
        HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, F32, F32), \
        SOURCE },

#define COMPARISONS_KERNELS_F32_2D(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 1), \
        HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, F32, F32), \
        SOURCE },

#define LESS_OP              less
#define GREAT_OP             great
#define LESS_EQUAL_OP        less_equal
#define GREAT_EQUAL_OP       great_equal
#define EQUAL_OP             equal
#define NOT_EQUAL_OP         not_equal

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } _comparisons_cl_kernel_map[] =
{
    COMPARISONS_KERNELS_F32(LESS_OP, COMP_LESS, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_F32(LESS_OP, COMP_LESS, F32,   F32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_U32(LESS_OP, COMP_LESS, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(LESS_OP, COMP_LESS, I32,   I32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(LESS_OP, COMP_LESS, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_F32_2D(LESS_OP, COMP_LESS, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_F32_2D(LESS_OP, COMP_LESS, F32,   F32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_U32_2D(LESS_OP, COMP_LESS, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(LESS_OP, COMP_LESS, I32,   I32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(LESS_OP, COMP_LESS, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS_F32(GREAT_OP, COMP_GREAT, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_F32(GREAT_OP, COMP_GREAT, F32,   F32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_U32(GREAT_OP, COMP_GREAT, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(GREAT_OP, COMP_GREAT, I32,   I32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(GREAT_OP, COMP_GREAT, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_F32_2D(GREAT_OP, COMP_GREAT, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_F32_2D(GREAT_OP, COMP_GREAT, F32,   F32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_U32_2D(GREAT_OP, COMP_GREAT, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(GREAT_OP, COMP_GREAT, I32,   I32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(GREAT_OP, COMP_GREAT, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS_F32(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_F32(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F32,   F32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_U32(LESS_EQUAL_OP,  COMP_LESS_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(LESS_EQUAL_OP,  COMP_LESS_EQUAL, I32,   I32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(LESS_EQUAL_OP,  COMP_LESS_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_F32_2D(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_F32_2D(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F32,   F32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_U32_2D(LESS_EQUAL_OP,  COMP_LESS_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(LESS_EQUAL_OP,  COMP_LESS_EQUAL, I32,   I32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(LESS_EQUAL_OP,  COMP_LESS_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS_F32(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_F32(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F32,   F32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_U32(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I32,   I32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_F32_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_F32_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F32,   F32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_U32_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I32,   I32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS_F32(EQUAL_OP, COMP_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_F32(EQUAL_OP, COMP_EQUAL, F32,   F32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_U32(EQUAL_OP, COMP_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(EQUAL_OP, COMP_EQUAL, I32,   I32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(EQUAL_OP, COMP_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_F32_2D(EQUAL_OP, COMP_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_F32_2D(EQUAL_OP, COMP_EQUAL, F32,   F32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_U32_2D(EQUAL_OP, COMP_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(EQUAL_OP, COMP_EQUAL, I32,   I32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(EQUAL_OP, COMP_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS_F32(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_F32(NOT_EQUAL_OP, COMP_NOT_EQUAL, F32,   F32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_U32(NOT_EQUAL_OP, COMP_NOT_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(NOT_EQUAL_OP, COMP_NOT_EQUAL, I32,   I32,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_I32(NOT_EQUAL_OP, COMP_NOT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_F32_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_F32_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, F32,   F32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_U32_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, I32,   I32,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_I32_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)
};

#undef LESS_OP
#undef GREAT_OP
#undef LESS_EQUAL_OP
#undef GREAT_EQUAL_OP
#undef EQUAL_OP
#undef NOT_EQUAL_OP

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_INPUT0_SCALE          (3)
#define SCALAR_INPUT0_TAIL           (4)
#define SCALAR_INPUT1_SCALE          (5)
#define SCALAR_INPUT1_TAIL           (6)
#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_comparisons_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_shape  = attr[2]->shape;

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
} /* _comparisons_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t operation,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype;
    vsi_nn_kernel_dtype_e input1_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    size_t i;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_NONE && input0_dtype == I16)
    {
        input0_dtype = I32;
    }

    if (inputs[1]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_NONE && input1_dtype == I16)
    {
        input1_dtype = I32;
    }

    if (outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_NONE && output_dtype == I8)
    {
        output_dtype = BOOL8;
    }

    key = HASH_COMPARISONS_KEY( operation, input0_dtype, input1_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(_comparisons_cl_kernel_map); i ++ )
    {
        if( _comparisons_cl_kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(_comparisons_cl_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _comparisons_cl_kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _comparisons_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                _comparisons_cl_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _comparisons_cl_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    int32_t operation = vsi_nn_kernel_param_get_int32( params, "operation" );
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    vsi_size_t new_rank = 0;
    vsi_bool ret = FALSE;

    float input0Scale = vsi_nn_get_tensor_scale(inputs[0]);
    float input0Tail = (float)vsi_nn_get_tensor_zero_point(inputs[0]) * input0Scale;
    float input1Scale = vsi_nn_get_tensor_scale(inputs[1]);
    float input1Tail = (float)vsi_nn_get_tensor_zero_point(inputs[1]) * input1Scale;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    ret = vsi_nn_kernel_optimize_eltwise_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            inputs[1]->attr.size, inputs[1]->attr.dim_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    if ( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
                inputs[1], shapes[1], new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
                outputs[0], shapes[2], new_rank );

#define _swap_tensor(a, b, tmp)  \
    { \
        tmp = a; \
        a = b; \
        b = tmp; \
    }

        if (shapes[1][3] > shapes[0][3] && new_rank == 4)
        {
            vsi_nn_tensor_t* reshape_tmp;
            _swap_tensor(reshape_tensors[0], reshape_tensors[1], reshape_tmp);

            if (VSI_NN_RELATIONAL_OPS_GREAT == operation)
            {
                operation = VSI_NN_RELATIONAL_OPS_LESS;
            }
            else if (VSI_NN_RELATIONAL_OPS_LESS == operation)
            {
                operation = VSI_NN_RELATIONAL_OPS_GREAT;
            }
            else if (VSI_NN_RELATIONAL_OPS_GREAT_EQUAL == operation)
            {
                operation = VSI_NN_RELATIONAL_OPS_LESS_EQUAL;
            }
            else if (VSI_NN_RELATIONAL_OPS_LESS_EQUAL == operation)
            {
                operation = VSI_NN_RELATIONAL_OPS_GREAT_EQUAL;
            }
        }

#undef _swap_tensor
    }
    else
    {
        goto final;
    }

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[2]->attr.size,
                reshape_tensors[2]->attr.dim_num ) )
    {
        goto final;
    }

    image_2d = (reshape_tensors[2]->attr.dim_num == 2 || reshape_tensors[2]->attr.size[2] == 1);
    status = _query_kernel( reshape_tensors, &reshape_tensors[2], operation, image_2d, kernel );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if ( node )
        {
            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    reshape_tensors, 2, &reshape_tensors[2], 1 );
            node_params[SCALAR_INPUT0_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input0Scale );
            node_params[SCALAR_INPUT0_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input0Tail );
            node_params[SCALAR_INPUT1_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input1Scale );
            node_params[SCALAR_INPUT1_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &input1Tail );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT0_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT0_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT1_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT1_TAIL] );
        }
    }

final:
    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );
    vsi_safe_release_tensor( reshape_tensors[2] );

    return node;
} /* _setup() */


REGISTER_BACKEND_CL( relational_ops, _setup )

__END_DECLS
#endif
