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

#define MOD_HASH_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

 #define MOD_KERNEL_SOURCE_NAME "mod"

#define MOD_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { MOD_HASH_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
      CVIVANTE_NAMESPACE("evis.mod_"#IN0_TYPE#IN1_TYPE"to"#OUT_TYPE), \
      MOD_KERNEL_SOURCE_NAME },

#define MOD_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { MOD_HASH_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
      CVIVANTE_NAMESPACE("evis.mod_"#IN0_TYPE#IN1_TYPE"to"#OUT_TYPE"_2D"), \
      MOD_KERNEL_SOURCE_NAME },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _mod_kernel_map[] =
{
        // Register kernel here
    MOD_KERNELS( F16,  F16,  F16 )
    MOD_KERNELS( F16,  F16,  I16 )
    MOD_KERNELS( F16,  F16,  I8 )
    MOD_KERNELS( F16,  F16,  U8 )
    MOD_KERNELS( I16,  I16,  I16 )
    MOD_KERNELS( I8,   I8,   I8 )
    MOD_KERNELS( U8,   U8,   U8 )
    MOD_KERNELS( I16,  I16,  F16 )
    MOD_KERNELS( I8,   I8,   F16 )
    MOD_KERNELS( U8,   U8,   F16 )
    MOD_KERNELS( BF16, BF16, BF16 )

    MOD_KERNELS_2D( F16,  F16,  F16 )
    MOD_KERNELS_2D( F16,  F16,  I16 )
    MOD_KERNELS_2D( F16,  F16,  I8 )
    MOD_KERNELS_2D( F16,  F16,  U8 )
    MOD_KERNELS_2D( I16,  I16,  I16 )
    MOD_KERNELS_2D( I8,   I8,   I8 )
    MOD_KERNELS_2D( U8,   U8,   U8 )
    MOD_KERNELS_2D( I16,  I16,  F16 )
    MOD_KERNELS_2D( I8,   I8,   F16 )
    MOD_KERNELS_2D( U8,   U8,   F16 )
    MOD_KERNELS_2D( BF16, BF16, BF16 )
};


/*
 * Kernel params
 */
static vx_param_description_t _mod_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MOD_PARAM_NUM  _cnt_of_array( _mod_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_mod_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vx_status     status             = VX_FAILURE;
    vx_tensor     input0              = (vx_tensor)param[0];
    vx_tensor     input1              = (vx_tensor)param[1];
    vx_tensor     output              = (vx_tensor)param[2];
    vsi_nn_kernel_tensor_attr_t *input0_attr  = NULL;
    vsi_nn_kernel_tensor_attr_t *input1_attr  = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr  = NULL;
    vsi_size_array_t             *output_shape = NULL;
    vsi_nn_kernel_dtype_e        input0_dtype = F16;
    int32_t                      input0_fl    = 0;
    int32_t                      input1_fl    = 0;
    int32_t                      output_fl    = 0;
    float                        inScale0     = 1.0f;
    float                        inScale1     = 1.0f;
    float                        outScale     = 1.0f;
    float                        in0Tail      = 0;
    float                        in1Tail      = 0;
    float                        outZp        = 0;

    input0_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input0 );
    CHECK_PTR_FAIL_GOTO( input0_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    input1_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input1 );
    CHECK_PTR_FAIL_GOTO( input1_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output );
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_shape = output_attr->shape;
    input0_dtype = input0_attr->dtype;

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

    if (input0_attr->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        input0_fl = input0_attr->dfp.fl;
        if (input0_fl > 0)
        {
            inScale0 = 1.0f / (float) ((int64_t)1 << input0_fl);
        }
        else
        {
            inScale0 = (float)((int64_t)1 << -input0_fl);
        }
    }
    else if (input0_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        inScale0   = input0_attr->asymm.scale;
        in0Tail    = -inScale0 * ((float)input0_attr->asymm.zero_point);
    }

    if (input1_attr->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        input1_fl = input1_attr->dfp.fl;
        if (input1_fl > 0)
        {
            inScale1 = 1.0f / (float) ((int64_t)1 << input1_fl);
        }
        else
        {
            inScale1 = (float)((int64_t)1 << -input1_fl);
        }
    }
    else if (input1_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        inScale1   = input1_attr->asymm.scale;
        in1Tail    = -inScale1 * ((float)input1_attr->asymm.zero_point);
    }

    if (output_attr->quant == VSI_NN_KERNEL_QUANT_DFP)
    {
        output_fl = output_attr->dfp.fl;
        if (output_fl > 0)
        {
            outScale = (float) ((int64_t)1 << output_fl);
        }
        else
        {
            outScale = 1.0f / (float)((int64_t)1 << -output_fl);
        }
    }
    else if (output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        outScale    = 1.0f / output_attr->asymm.scale;
        outZp       = (float)(output_attr->asymm.zero_point);
    }

    if (BF16 == input0_dtype)
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
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        status  = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniConvertFstToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniConvertSecToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertFstToFp32_4x4", &uniConvertFstToFp32_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertSecToFp32_4x4", &uniConvertSecToFp32_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node, "in_scale0", &inScale0 );
        status |= vsi_nn_kernel_gpu_add_param( node, "in0Tail", &in0Tail );
        status |= vsi_nn_kernel_gpu_add_param( node, "in_scale1", &inScale1 );
        status |= vsi_nn_kernel_gpu_add_param( node, "in1Tail", &in1Tail );
        status |= vsi_nn_kernel_gpu_add_param( node, "out_scale", &outScale );
        status |= vsi_nn_kernel_gpu_add_param( node, "out_zp", &outZp );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (input0_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input0_attr);
    }
    if (input1_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input1_attr);
    }
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }
    return status;
} /* _mod_initializer() */



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
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _mod_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _mod_kernel_map );
    vx_param_description_t * param_def  = _mod_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _mod_kernel_param_def );
    vx_kernel_initialize_f  initializer = _mod_initializer;

    uint32_t key = 0;
    uint32_t i = 0;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = MOD_HASH_KEY( in0_dtype, in1_dtype, out_dtype, image_2d);

    for (i = 0; i < kernel_map_size; i ++)
    {
        if (kernel_map[i].key == key)
        {
            break;
        }
    }

    if (i < kernel_map_size)
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
    vsi_nn_kernel_node_param_t node_params[_MOD_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    int32_t isfmod = vsi_nn_kernel_param_get_int32(params, "isfmod");

    if (!vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size, outputs[0]->attr.dim_num ))
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2 || outputs[0]->attr.size[2] == 1);
    if (vsi_nn_kernel_map_dtype(inputs[0]->attr.dtype.vx_type) == F16 ||
        vsi_nn_kernel_map_dtype(inputs[1]->attr.dtype.vx_type) == F16 ||
        vsi_nn_kernel_map_dtype(inputs[0]->attr.dtype.vx_type) == BF16 ||
        vsi_nn_kernel_map_dtype(inputs[1]->attr.dtype.vx_type) == BF16)
    {
        isfmod = 1;
    }
    status = _query_kernel( kernel, inputs, outputs, image_2d);
    if (VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if (node)
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _MOD_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[3] = vsi_nn_kernel_scalar_create( graph, I32, &isfmod );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _MOD_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[3] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( mod, _setup )

