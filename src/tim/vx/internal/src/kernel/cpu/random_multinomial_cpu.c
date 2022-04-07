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
#include <float.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _CPU_ARG_NUM            (0)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("random_multinomial_sw")

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _PARAM_NUM  _cnt_of_array( kernel_param_def )

/*
 * Kernel function
 */
static int upper_bound(float* a, int n, float x) {
    int l = 0;
    int h = n;
    while (l < h) {
        int mid = (l + h) / 2;
        if (x >= a[mid]) {
            l = mid + 1;
        } else {
            h = mid;
        }
    }
    return l;
} /* upper_bound() */

DEF_KERNEL_EXECUTOR(_compute)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    vsi_size_t out_elements = 0;
    vsi_size_t stride_size[_CPU_INPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t *random_integer = NULL;
    float *random_float = NULL;
    float *cdf = NULL;
    uint32_t i = 0;
    uint32_t n = 0;
    uint32_t batch = 0;
    uint32_t class_size = 0;
    int32_t sample_num = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    sample_num = (int32_t)attr[2]->shape->data[0];
    batch = (int32_t)attr[0]->shape->data[1];
    class_size = (int32_t)attr[0]->shape->data[0];

    vsi_nn_kernel_tensor_attr_get_stride( attr[0], stride_size[0] );
    vsi_nn_kernel_tensor_attr_get_stride( attr[1], stride_size[1] );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[2] );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input1 buffer fail.", final );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset( buffer[2], 0, out_elements * sizeof(float) );

    random_integer = (uint32_t *)malloc(out_elements * sizeof(uint32_t));
    CHECK_PTR_FAIL_GOTO( random_integer, "Create buffer fail.", final );
    random_float = (float *)malloc(out_elements * sizeof(float));
    CHECK_PTR_FAIL_GOTO( random_float, "Create buffer fail.", final );
    cdf = (float *)malloc(class_size * sizeof(float));
    CHECK_PTR_FAIL_GOTO( cdf, "Create buffer fail.", final );

    vsi_nn_random_init_for_philox_4x32_10((uint32_t)(buffer[1][0]),
        (uint32_t)(buffer[1][1]));
    vsi_nn_random_generate_by_philox_4x32_10(random_integer, (uint32_t)out_elements);
    vsi_nn_random_uniform_transform(random_integer,
            random_float, (uint32_t)out_elements);

    for (n = 0; n < batch; n++)
    {
        uint32_t c = 0;
        float batch_max = -FLT_MAX;
        float total = 0;
        for(c = 0; c < class_size; c++)
        {
            uint32_t index = n * class_size + c;
            batch_max = vsi_nn_max(batch_max, buffer[0][index]);
        }

        for(c = 0; c < class_size; c++)
        {
            uint32_t index = n * class_size + c;
            total += (float)(exp(buffer[0][index] - batch_max));
            cdf[c] = total;
        }

        for(c = 0; c < (uint32_t)sample_num; c++)
        {
            uint32_t index = n * sample_num + c;
            float target = random_float[index] * total;
            uint32_t out_class = upper_bound(cdf, class_size, target);
            buffer[2][index] = (float)out_class;
        }
    }
    status = vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
            buffer[2], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
    }

    if (cdf)
    {
        free(cdf);
        cdf = NULL;
    }
    if (random_integer)
    {
        free(random_integer);
        random_integer = NULL;
    }
    if (random_float)
    {
        free(random_float);
        random_float = NULL;
    }

    return status;
} /* _compute() */

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
    kernel->info.function    = _compute;
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
    vsi_nn_kernel_node_param_t node_params[_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( random_multinomial, _setup )
