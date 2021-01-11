/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_U8TOU8       CVIVANTE_NAMESPACE("evis.depth2space_crd_U8toU8")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_I8TOI8       CVIVANTE_NAMESPACE("evis.depth2space_crd_I8toI8")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_I16TOI16     CVIVANTE_NAMESPACE("evis.depth2space_crd_I16toI16")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_F16TOF16     CVIVANTE_NAMESPACE("evis.depth2space_crd_F16toF16")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_I8TOF16      CVIVANTE_NAMESPACE("evis.depth2space_crd_I8toF16")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_I16TOF16     CVIVANTE_NAMESPACE("evis.depth2space_crd_I16toF16")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_F16TOI8      CVIVANTE_NAMESPACE("evis.depth2space_crd_F16toI8")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_F16TOI16     CVIVANTE_NAMESPACE("evis.depth2space_crd_F16toI16")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_U8TOF16      CVIVANTE_NAMESPACE("evis.depth2space_crd_U8toF16")
#define VX_KERNEL_NAME_DEPTH2SPACE_CRD_F16TOU8      CVIVANTE_NAMESPACE("evis.depth2space_crd_F16toU8")

#define KERNEL_SOURCE_1    "depth2space_crd"

// Add kernel hashtable here
#define HASH_DEPTH2SPACE_CRD_KEY(_input0_type, _output_type, _quant_type) \
    ((_input0_type << 24) | (_output_type << 16) | (_quant_type << 8))

#define TENSOR_DEPTH2SPACE_CRD_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_DEPTH2SPACE_CRD_KEY(IN0_TYPE, OUT_TYPE, 0), \
        VX_KERNEL_NAME_DEPTH2SPACE_CRD_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } depth2space_crd_map[] =
{
    TENSOR_DEPTH2SPACE_CRD_KERNELS(U8,  U8,        KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(I8,  I8,        KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(I16, I16,       KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(F16, F16,       KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(I8,  F16,       KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(I16, F16,       KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(F16, I8,        KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(F16, I16,       KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(U8,  F16,       KERNEL_SOURCE_1)
    TENSOR_DEPTH2SPACE_CRD_KERNELS(F16, U8,        KERNEL_SOURCE_1)
};

/*
 * Kernel params
 */
static vx_param_description_t _depth2space_crd_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _DEPTH2SPACE_CRD_PARAM_NUM  _cnt_of_array( _depth2space_crd_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_depth2space_crd_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    uint32_t      output_dims = 0;
    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    int32_t     output_width  = 0;
    int32_t     output_height = 0;
    int32_t     output_chn = 0;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 0;
    int32_t     dstZP      = 0;
    float       dstScale   = 0;

    uint32_t pack_key = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    src0ZP     = attr[0]->asymm.zero_point;
    src0Scale  = attr[0]->asymm.scale;
    dstZP      = attr[1]->asymm.zero_point;
    dstScale   = attr[1]->asymm.scale;
    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            src0Scale = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            src0Scale = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        src0Scale = 1;
    }

    if( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[1]->dfp.fl > 0)
        {
            dstScale = (float)((int64_t)1 << attr[1]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)((int64_t)1 << -attr[1]->dfp.fl));
        }
        dstScale = 1.0f/dstScale;
    }
    else if( attr[1]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        dstScale = 1;
    }

    output_dims = (uint32_t)attr[1]->shape->size;
    output_width = attr[1]->shape->data[0];
    output_height = attr[1]->shape->data[1];
    output_chn = output_dims > 2 ? attr[1]->shape->data[2] : 1;

    shaderParam.global_scale[0]  = 1;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((output_width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = output_height;
    shaderParam.global_size[2]   = output_chn;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE)    \
        (IN0_TYPE | (OUT_TYPE << 8))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype);

    {
        uint16_t M0               = 0;
        int32_t  postShift        = 0;
        uint32_t multAndoutZP0[2] = {0};
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8, F16):
        case _PACK_SELECT_KEY( I8, F16):
        case _PACK_SELECT_KEY( I16, F16):
        case _PACK_SELECT_KEY( F16, U8):
        case _PACK_SELECT_KEY( F16, I8):
        case _PACK_SELECT_KEY( F16, I16):
        case _PACK_SELECT_KEY( U8, U8):
        case _PACK_SELECT_KEY( I8, I8):
        case _PACK_SELECT_KEY( I16, I16):
            {
                gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift);
                multAndoutZP0[0] = (uint32_t)(M0);
                multAndoutZP0[1] = (uint32_t)((dstZP << postShift) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift );
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }
#undef _PACK_SELECT_KEY

    CHECK_STATUS_FAIL_GOTO(status, OnError );

OnError:
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
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_DEPTH2SPACE_CRD_KEY( input0_dtype, output_dtype, 0 );

    for( i = 0; i < _cnt_of_array(depth2space_crd_map); i ++ )
    {
        if( depth2space_crd_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(depth2space_crd_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  depth2space_crd_map[i].function_name );
        kernel->info.parameters = _depth2space_crd_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _depth2space_crd_kernel_param_def );
        kernel->info.initialize = _depth2space_crd_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                depth2space_crd_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                depth2space_crd_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_DEPTH2SPACE_CRD_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, params );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            vsi_nn_kernel_node_pack_io( tmp_params, _DEPTH2SPACE_CRD_PARAM_NUM, inputs, 1, outputs, 1 );
            tmp_params[2] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _DEPTH2SPACE_CRD_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[2] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( depth2space_internal, _setup )

