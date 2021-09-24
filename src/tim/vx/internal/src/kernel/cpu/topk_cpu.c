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

/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (2)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.topk")


/*
 * Kernel params
 */
static vx_param_description_t _topk_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    // Add kererl parameters here
};
#define _TOPK_PARAM_NUM  _cnt_of_array( _topk_kernel_param_def )

static uint32_t _max_comp_func(void* data, int32_t left, int32_t right)
{
    float* fdata = (float*)data;
    if (fdata[left] >= fdata[right])
    {
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

static void _find_top_k_1d
(
    float* input,
    uint32_t input_len,
    uint32_t k,
    float* value,
    uint32_t* indices
)
{
    int32_t low = 0;
    int32_t high = input_len - 1;
    int32_t j;

    for (j = 0; j < (int32_t)input_len; j++)
    {
        indices[j] = j;
    }

    j = vsi_nn_partition(input, low, high, _max_comp_func, FALSE, indices);

    //part_sort
    while (j != (int32_t)k)
    {
        if ((int32_t)k > j)
        {
            low = j + 1;
        }
        else
        {
            high = j;
        }
        j = vsi_nn_partition(input, low, high, _max_comp_func, FALSE, indices);
    }
    //all_sort
    vsi_nn_partition(input, 0, k - 1, _max_comp_func, TRUE, indices);

    for (j = 0; j < (int32_t)k; j++)
    {
        value[j] = input[indices[j]];
    }
}

/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_compute)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t input[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float *f32_in_buffer[_INPUT_NUM] = {NULL};
    float *f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *in_attr[_INPUT_NUM];
    vsi_nn_kernel_tensor_attr_t *out_attr[_OUTPUT_NUM];
    vsi_size_t   out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t   out_elements[_OUTPUT_NUM] = {0};
    vsi_size_t   out_bytes[_OUTPUT_NUM] = {0};
    uint32_t  i = 0;
    int32_t  j = 0;
    int32_t  top_k = 0;
    uint32_t block_num = 0;
    uint32_t block_size = 0;
    uint32_t * indices_ptr = NULL;

    /* prepare data */
    for (i = 0; i < _INPUT_NUM; i ++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create( input[i] );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( input[i], in_attr[i], TRUE );
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create input0 buffer fail.", final );
    }

    for (i = 0; i < _OUTPUT_NUM; i ++)
    {
        output[i] = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        out_attr[i] = vsi_nn_kernel_tensor_attr_create( output[i] );
        vsi_nn_kernel_tensor_attr_get_stride( out_attr[i], out_stride_size[i] );
        out_elements[i] = vsi_nn_kernel_tensor_attr_get_size( out_attr[i] );
        out_bytes[i] = out_elements[i] * sizeof(float);
        f32_out_buffer[i] = (float *)malloc( out_bytes[i] );
        CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create output buffer fail.", final );
        memset( f32_out_buffer[i], 0, out_bytes[i] );
    }

    status = vsi_nn_kernel_scalar_read_int32( param[3], &top_k );
    CHECK_STATUS_FAIL_GOTO(status, final );

    block_num = (uint32_t)in_attr[0]->shape->data[1];
    block_size = (uint32_t)in_attr[0]->shape->data[0];
    indices_ptr = (uint32_t*)malloc(block_size * sizeof(uint32_t));
    CHECK_PTR_FAIL_GOTO( indices_ptr, "Create indices buffer fail.", final );

    for(i = 0; i < block_num; i++)
    {
        uint32_t in_index = i * block_size;
        uint32_t out_index = i * top_k;
        _find_top_k_1d(&(f32_in_buffer[0][in_index]),
            block_size, top_k, &(f32_out_buffer[0][out_index]), indices_ptr);

        for (j = 0; j < top_k; j++)
        {
            f32_out_buffer[1][out_index + j] = (float)indices_ptr[j];
        }
    }
    // Handle the 1D input
    if (!block_num)
    {
        _find_top_k_1d(&(f32_in_buffer[0][0]),
            block_size, top_k, &(f32_out_buffer[0][0]), indices_ptr);
        for (j = 0; j < top_k; j++)
        {
            f32_out_buffer[1][j] = (float)indices_ptr[j];
        }
    }

    /* save data */
    for(i = 0; i < _OUTPUT_NUM; i++)
    {
        status = vsi_nn_kernel_tensor_write_from_float( output[i], out_attr[i],
                f32_out_buffer[i], out_elements[i] );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    vsi_nn_safe_free(indices_ptr);
    for (i = 0; i < _INPUT_NUM; i++)
    {
        if (f32_in_buffer[i])
        {
            free(f32_in_buffer[i]);
            f32_in_buffer[i] = NULL;
        }
        if (in_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &in_attr[i] );
        }
    }
    for (i = 0; i < _OUTPUT_NUM; i++)
    {
        if (f32_out_buffer[i])
        {
            free(f32_out_buffer[i]);
            f32_out_buffer[i] = NULL;
        }
        if (out_attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &out_attr[i] );
        }
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
    kernel->info.parameters  = _topk_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _topk_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_TOPK_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t top_k = vsi_nn_kernel_param_get_int32(params, "top_k");

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _TOPK_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[3] = vsi_nn_kernel_scalar_create( graph, I32, &top_k );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _TOPK_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &node_params[3] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( topk, _setup )

