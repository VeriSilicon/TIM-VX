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
#include "kernel/vsi_nn_kernel.h"
#include "client/vsi_nn_vxkernel.h"

__BEGIN_DECLS

#define _CPU_ARG_NUM            (10)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.pre_process_nv12_sw")

#define DESCALE(x) (((x) + (1<<19)) >> 20)

DEF_KERNEL_EXECUTOR(_pre_process_nv12_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    float * outBuffer = NULL;
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t xRatio = 0, yRatio = 0, xOffset = 0, yOffset = 0;
    float rMean = 0, gMean = 0, bMean = 0, var = 0;
    int32_t order = 0, trans = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[2] );

    i = 3;
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xRatio);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yRatio);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xOffset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yOffset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &rMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &gMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &bMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &var);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &order);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &trans);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input1 buffer fail.", final );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset( buffer[2], 0, out_elements * sizeof(float) );

    if(trans)
    {
        outBuffer = (float *)malloc( out_elements * sizeof(float) );
        CHECK_PTR_FAIL_GOTO( outBuffer, "Create output buffer fail.", final );
        memset( outBuffer, 0, out_elements * sizeof(float) );
    }

    {
        int32_t dx, dy, dz;
        int32_t src_width = attr[0]->shape->data[0];
        int32_t src_height = attr[0]->shape->data[1];
        int32_t dst_width = trans ? attr[2]->shape->data[1] : attr[2]->shape->data[0];
        int32_t dst_height = trans ? attr[2]->shape->data[2] : attr[2]->shape->data[1];
        int32_t stride = dst_width * dst_height;
        int32_t rOffset = 0;
        int32_t gOffset = 1 * stride;
        int32_t bOffset = 2 * stride;
        float D, E;
        float R, G, B;
        float min = 0;
        float max = 255;
        float* src_y_slice = NULL;
        float* src_uv_yScanline = NULL;


        uint32_t xrIntFloat_16 = (src_width << 16) / dst_width + 1;
        uint32_t yrIntFloat_16 = (src_height << 16) / dst_height + 1;
        uint32_t srcy = 0, srcx = 0;

        if(attr[2]->dtype == I8)
        {
            min = -128;
            max = 127;
        }
        else if(attr[2]->dtype == I16 || attr[2]->dtype == F16)
        {
            min = -65536;
            max = 65535;
        }

        if(order)
        {
            rOffset = 2 * stride;
            bOffset = 0;
        }

        for ( dz = 0; dz < 1; dz ++)
        {
            for ( dy = 0; dy < (int32_t)dst_height; dy ++)
            {
                srcy = (((uint32_t)dy * yrIntFloat_16) >> 16) + yOffset;
                src_y_slice = buffer[0] + (srcy) * src_width;
                src_uv_yScanline = buffer[1] + (srcy / 2) * src_width;

                for ( dx = 0; dx < (int32_t)dst_width; dx ++)
                {
                    float finalVal = 0;
                    int32_t output_index = 0;
                    int32_t dstR_idx = 0, dstG_idx = 0, dstB_idx = 0;
                    float tmpY = 0.0f;
                    float tmpU = 0.0f;
                    float tmpV = 0.0f;

                    srcx = (((uint32_t)dx * xrIntFloat_16) >> 16) + xOffset;
                    tmpY = src_y_slice[srcx];
                    tmpU = src_uv_yScanline[(srcx / 2) * 2];
                    tmpV = src_uv_yScanline[(srcx / 2) * 2 + 1];

                    D = (tmpU - 128);
                    E = (tmpV - 128);

                    // B
                    B = (float)vsi_clamp((tmpY + (1.7790 * D)), min, max);
                    //G
                    G = (float)vsi_clamp((tmpY - 0.3455 * D - 0.7169 * E), min, max);
                    //R
                    R = (float)vsi_clamp((tmpY + 1.4065 * E), min, max);

                    output_index = dx + dy * dst_width;

                    dstR_idx = output_index + rOffset;
                    dstG_idx = output_index + gOffset;
                    dstB_idx = output_index + bOffset;

                    finalVal = (B - bMean) * var;
                    buffer[2][dstB_idx] = finalVal;

                    finalVal = (G - gMean) * var;
                    buffer[2][dstG_idx] = finalVal;

                    finalVal = (R - rMean) * var;
                    buffer[2][dstR_idx] = finalVal;
                }
            }
        }
    }

    if(trans)
    {
        uint32_t shape[] = {attr[2]->shape->data[0], attr[2]->shape->data[1], attr[2]->shape->data[2], 1};
        uint32_t perm[] = {1, 2, 0, 3};
        vsi_nn_Transpose((uint8_t*)outBuffer, (uint8_t*)buffer[2],
                        shape, (uint32_t)attr[2]->shape->size, perm, VSI_NN_TYPE_FLOAT32);

        status = vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
            outBuffer, out_elements );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }
    else
    {
        status = vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
                buffer[2], out_elements );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    if(outBuffer)
    {
        free(outBuffer);
    }
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        if(attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }
    return status;
} /* _pre_process_nv12_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};


static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _pre_process_nv12_exec,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
    return VSI_SUCCESS;
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
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 3;
            int32_t scale_x  = vsi_nn_kernel_param_get_int32( params, "scale_x" );
            int32_t scale_y  = vsi_nn_kernel_param_get_int32( params, "scale_y" );
            int32_t left     = vsi_nn_kernel_param_get_int32( params, "left" );
            int32_t top      = vsi_nn_kernel_param_get_int32( params, "top" );
            float r_mean     = vsi_nn_kernel_param_get_float32( params, "r_mean" );
            float g_mean     = vsi_nn_kernel_param_get_float32( params, "g_mean" );
            float b_mean     = vsi_nn_kernel_param_get_float32( params, "b_mean" );
            float rgb_scale  = vsi_nn_kernel_param_get_float32( params, "rgb_scale" );
            int32_t reverse  = vsi_nn_kernel_param_get_int32( params, "reverse" );
            int32_t trans    = vsi_nn_kernel_param_get_int32( params, "enable_perm" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );

            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &rgb_scale );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &trans );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[3] );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
            vsi_nn_kernel_scalar_release( &backend_params[6] );
            vsi_nn_kernel_scalar_release( &backend_params[7] );
            vsi_nn_kernel_scalar_release( &backend_params[8] );
            vsi_nn_kernel_scalar_release( &backend_params[9] );
            vsi_nn_kernel_scalar_release( &backend_params[10] );
            vsi_nn_kernel_scalar_release( &backend_params[11] );
            vsi_nn_kernel_scalar_release( &backend_params[12] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( pre_process_nv12, _setup )

