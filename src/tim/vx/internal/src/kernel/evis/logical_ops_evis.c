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

__BEGIN_DECLS

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;

#define _LOGICAL_OPS_KERNEL_SOURCE      "logical_ops"

#define STR(a) #a

// Add kernel hashtable here
#define LOGICAL_OPS_HASH_KEY(OP_TYPE, IN_DTYPE, OUT_DTYPE, _image_2d) \
        ((OP_TYPE << 20) | ( IN_DTYPE << 12 ) | ( OUT_DTYPE << 4) | (_image_2d))

#define PACK_KERNEL_MAP(OP_TYPE, IN_DTYPE, OUT_DTYPE, op_name) \
        { LOGICAL_OPS_HASH_KEY(OP_TYPE, IN_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("evis.logical_"op_name"_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
        _LOGICAL_OPS_KERNEL_SOURCE}

#define PACK_KERNEL_MAP_2D(OP_TYPE, IN_DTYPE, OUT_DTYPE, op_name) \
        { LOGICAL_OPS_HASH_KEY(OP_TYPE, IN_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("evis.logical_"op_name"_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        _LOGICAL_OPS_KERNEL_SOURCE}

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _logical_ops_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP(VSI_NN_LOGICAL_OR,  I8,  I8,  "or"),
    PACK_KERNEL_MAP(VSI_NN_LOGICAL_AND, I8,  I8,  "and"),
    PACK_KERNEL_MAP(VSI_NN_LOGICAL_XOR, I8,  I8,  "xor"),
    PACK_KERNEL_MAP(VSI_NN_LOGICAL_OR,  BF16,  I8,  "or"),
    PACK_KERNEL_MAP(VSI_NN_LOGICAL_AND, BF16,  I8,  "and"),
    PACK_KERNEL_MAP(VSI_NN_LOGICAL_XOR, BF16,  I8,  "xor"),
    PACK_KERNEL_MAP_2D(VSI_NN_LOGICAL_OR,  I8,  I8,  "or"),
    PACK_KERNEL_MAP_2D(VSI_NN_LOGICAL_AND, I8,  I8,  "and"),
    PACK_KERNEL_MAP_2D(VSI_NN_LOGICAL_XOR, I8,  I8,  "xor"),
    PACK_KERNEL_MAP_2D(VSI_NN_LOGICAL_OR,  BF16,  I8,  "or"),
    PACK_KERNEL_MAP_2D(VSI_NN_LOGICAL_AND, BF16,  I8,  "and"),
    PACK_KERNEL_MAP_2D(VSI_NN_LOGICAL_XOR, BF16,  I8,  "xor"),
};


/*
 * Kernel params
 */
static vx_param_description_t _logical_ops_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _LOGICAL_OPS_PARAM_NUM  _cnt_of_array( _logical_ops_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_logical_ops_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VX_FAILURE;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vx_tensor     input            = (vx_tensor)param[0];
    vx_tensor     output           = (vx_tensor)param[2];
    vsi_nn_kernel_dtype_e        input_dtype = F16;
    vsi_nn_kernel_tensor_attr_t *input_attr = NULL, *output_attr = NULL;
    vsi_size_array_t             *output_shape  = NULL;

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input);
    CHECK_PTR_FAIL_GOTO( input_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_shape  = output_attr->shape;
    input_dtype   = input_attr->dtype;

    gpu_param.dim = output_shape->size < 3 ? 2 : 3;
    gpu_param.global_offset[0] = 0;
    gpu_param.global_offset[1] = 0;
    gpu_param.global_offset[2] = 0;
    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1)
                                             / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (output_shape->data[1] + gpu_param.global_scale[1] - 1)
                                             / gpu_param.global_scale[1];
    gpu_param.global_size[2]   = output_shape->size > 2 ?
                                 (output_shape->data[2] + gpu_param.global_scale[2] - 1)
                                             / gpu_param.global_scale[2] : 1;

    if(F16 == input_dtype)
    {
        gpu_dp_inst_t uniMulShortMinus1toFp16_2x8 = {{
            0x22222222, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node, "uniMulShortMinus1toFp16_2x8", &uniMulShortMinus1toFp16_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (BF16 == input_dtype)
    {
        gpu_dp_inst_t uniConvertInt16toInt8_2x8 = {{
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node, "uniConvertInt16toInt8_2x8", &uniConvertInt16toInt8_2x8);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (input_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input_attr);
    }
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }
    return status;
} /* _logical_ops_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d,
    vsi_nn_logical_ops_type_t op_type
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _logical_ops_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _logical_ops_kernel_map );
    vx_param_description_t * param_def  = _logical_ops_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _logical_ops_kernel_param_def );
    vx_kernel_initialize_f  initializer = _logical_ops_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (in_dtype != in1_dtype)
    {
        return VSI_FAILURE;
    }

    if (BOOL8 == in_dtype)
    {
        in_dtype  = I8;
    }

    if (BOOL8 == out_dtype)
    {
        out_dtype = I8;
    }

    key = LOGICAL_OPS_HASH_KEY(op_type, in_dtype, out_dtype, image_2d);

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
    vsi_nn_kernel_node_param_t node_params[_LOGICAL_OPS_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    uint32_t ops_type  = vsi_nn_kernel_param_get_int32( params, "ops_type" );

    if( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2);

    status = _query_kernel( kernel, inputs, outputs, image_2d, (vsi_nn_logical_ops_type_t)ops_type);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _LOGICAL_OPS_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _LOGICAL_OPS_PARAM_NUM );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( logical_ops, _setup )

