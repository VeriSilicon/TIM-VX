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

/*
 * Define kernel meta.
 */
#define _CPU_ARG_NUM            (11)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.pre_process_yuv422_sw")

#define DESCALE(x) (((x) + (1<<19)) >> 20)

static vx_param_description_t kernel_param_def[] =
{
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
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};


DEF_KERNEL_EXECUTOR(_pre_process_yuv422_exec)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    float * outBuffer = NULL;
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t xRatio = 0, yRatio = 0, xOffset = 0, yOffset = 0;
    float rMean = 0, gMean = 0, bMean = 0, var = 0;
    int32_t order = 0, trans = 0, yuv422_type = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    i = 2;
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
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yuv422_type);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    if(trans)
    {
        outBuffer = (float *)malloc( out_elements * sizeof(float) );
        CHECK_PTR_FAIL_GOTO( outBuffer, "Create output buffer fail.", final );
        memset( outBuffer, 0, out_elements * sizeof(float) );
    }

    {
        int32_t dx, dy, dz;
        int32_t src_width = (int32_t)attr[0]->shape->data[0];
        int32_t dst_width = (int32_t)(trans ? attr[1]->shape->data[1] : attr[1]->shape->data[0]);
        int32_t dst_height = (int32_t)(trans ? attr[1]->shape->data[1] : attr[1]->shape->data[1]);
        int32_t stride = (int32_t)(dst_width * dst_height);
        int32_t rOffset = 0;
        int32_t gOffset = 1 * stride;
        int32_t bOffset = 2 * stride;
        float D0, D1, E0, E1;
        float R0, G0, B0, R1, G1, B1;
        float min = 0;
        float max = 255;
        float* src_y_slice = NULL;

        uint32_t roi_width = (xRatio * dst_width) >> 15;
        uint32_t roi_height = (yRatio * dst_height) >> 15;
        uint32_t xrIntFloat_16 = (roi_width << 16) / dst_width + 1;
        uint32_t yrIntFloat_16 = (roi_height << 16) / dst_height + 1;
        uint32_t srcy = 0, srcx = 0;

        if(attr[1]->dtype == I8)
        {
            min = -128;
            max = 127;
        }
        else if(attr[1]->dtype == I16 || attr[1]->dtype == F16)
        {
            min = -32768;
            max = 32767;
        }

        if(order)
        {
            rOffset = 2 * stride;
            bOffset = 0;
        }

        for ( dz = 0; dz < 1; dz ++)
        {
            for ( dy = 0; dy < (int32_t)dst_height; dy++)
            {
                srcy = (((uint32_t)dy * yrIntFloat_16) >> 16) + yOffset;
                src_y_slice = buffer[0] + (srcy) * src_width;
                for ( dx = 0; dx < (int32_t)dst_width; dx += 2)
                {
                    int32_t output_index = 0;
                    int32_t dstR_idx = 0, dstG_idx = 0, dstB_idx = 0;
                    float tmpY0 = 0.0f;
                    float tmpY1 = 0.0f;
                    float tmpU0 = 0.0f;
                    float tmpU1 = 0.0f;
                    float tmpV0 = 0.0f;
                    float tmpV1 = 0.0f;

                    srcx = ((((uint32_t)dx * xrIntFloat_16) >> 16) + xOffset) * 2;

                    if (xrIntFloat_16 >> 16 == 1)
                    {
                        if (yuv422_type == 1)
                        {
                            tmpY0 = src_y_slice[srcx + 1];
                            tmpU0 = src_y_slice[srcx];
                            tmpY1 = src_y_slice[srcx + 3];
                            tmpV0 = src_y_slice[srcx + 2];
                            tmpU1 = tmpU0;
                            tmpV1 = tmpV0;
                        }
                        else
                        {
                            tmpY0 = src_y_slice[srcx];
                            tmpU0 = src_y_slice[srcx + 1];
                            tmpY1 = src_y_slice[srcx + 2];
                            tmpV0 = src_y_slice[srcx + 3];
                            tmpU1 = tmpU0;
                            tmpV1 = tmpV0;
                        }
                    }
                    else
                    {
                        if (yuv422_type == 1)
                        {
                            tmpY0 = src_y_slice[srcx + 1];
                            tmpU0 = src_y_slice[(srcx / 4) * 4];
                            tmpV0 = src_y_slice[(srcx / 4) * 4 + 2];
                            srcx = (((uint32_t)(dx + 1) * xrIntFloat_16) >> 16) + xOffset;
                            srcx = srcx * 2;
                            tmpY1 = src_y_slice[srcx + 1];
                            tmpU1 = src_y_slice[(srcx / 4) * 4];
                            tmpV1 = src_y_slice[(srcx / 4) * 4 + 2];
                        }
                        else
                        {
                            tmpY0 = src_y_slice[srcx];
                            tmpU0 = src_y_slice[(srcx / 4) * 4 + 1];
                            tmpV0 = src_y_slice[(srcx / 4) * 4 + 3];
                            srcx = (((uint32_t)(dx + 1) * xrIntFloat_16) >> 16) + xOffset;
                            srcx = srcx * 2;
                            tmpY1 = src_y_slice[srcx];
                            tmpU1 = src_y_slice[(srcx / 4) * 4 + 1];
                            tmpV1 = src_y_slice[(srcx / 4) * 4 + 3];
                        }
                    }

                    D0 = (tmpU0 - 128);
                    E0 = (tmpV0 - 128);
                    D1 = (tmpU1 - 128);
                    E1 = (tmpV1 - 128);

                    B0 = (float)vsi_clamp((tmpY0 + (1.7790 * D0)), min, max);
                    G0 = (float)vsi_clamp((tmpY0 - 0.3455 * D0 - 0.7169 * E0), min, max);
                    R0 = (float)vsi_clamp((tmpY0 + 1.4065 * E0), min, max);

                    B1 = (float)vsi_clamp((tmpY1 + (1.7790 * D1)), min, max);
                    G1 = (float)vsi_clamp((tmpY1 - 0.3455 * D1 - 0.7169 * E1), min, max);
                    R1 = (float)vsi_clamp((tmpY1 + 1.4065 * E1), min, max);

                    output_index = dx + dy * dst_width;

                    dstR_idx = output_index + rOffset;
                    dstG_idx = output_index + gOffset;
                    dstB_idx = output_index + bOffset;

                    buffer[1][dstB_idx] = (B0 - bMean) * var;
                    buffer[1][dstG_idx] = (G0 - gMean) * var;
                    buffer[1][dstR_idx] = (R0 - rMean) * var;

                    dstR_idx += 1;
                    dstG_idx += 1;
                    dstB_idx += 1;

                    buffer[1][dstB_idx] = (B1 - bMean) * var;
                    buffer[1][dstG_idx] = (G1 - gMean) * var;
                    buffer[1][dstR_idx] = (R1 - rMean) * var;
                }
            }
        }
    }

    if(trans)
    {
        vsi_size_t shape[] = {attr[1]->shape->data[0], attr[1]->shape->data[1], attr[1]->shape->data[2], 1};
        vsi_size_t perm[] = {1, 2, 0, 3};
        vsi_nn_Transpose((uint8_t*)outBuffer, (uint8_t*)buffer[1],
                        shape, (uint32_t)attr[1]->shape->size, perm, VSI_NN_TYPE_FLOAT32);

        status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            outBuffer, out_elements );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }
    else
    {
        status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
                buffer[1], out_elements );
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
} /* _pre_process_yuv422_exec() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _pre_process_yuv422_exec;
    kernel->info.parameters  = kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( kernel_param_def );
    status = VSI_SUCCESS;
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
    vsi_nn_kernel_node_param_t node_params[_CPU_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    status = _query_kernel( kernel, inputs, outputs);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 2;
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
            int32_t yuv422_type = vsi_nn_kernel_param_get_int32( params, "yuv422_type" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );

            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &rgb_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &trans );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &yuv422_type );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( pre_process_yuv422, _setup )

