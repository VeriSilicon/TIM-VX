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
#ifndef _VSI_NN_OP_CLIENT_STRIDED_SLICE_H
#define _VSI_NN_OP_CLIENT_STRIDED_SLICE_H

#include "vsi_nn_platform.h"
#include "vsi_nn_types.h"
#include "utils/vsi_nn_link_list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _strided_slice_param
{
    int32_t *begin_dims;
    int32_t begin_dims_num;
    int32_t *end_dims;
    int32_t end_dims_num;
    int32_t *stride_dims;
    int32_t stride_dims_num;
    int32_t begin_mask;
    int32_t end_mask;
    int32_t shrink_axis_mask;
    int32_t new_axis_mask;

    int32_t num_add_axis;
} strided_slice_param;

typedef struct _vsi_nn_strided_slice_lcl_data2
{
    vsi_nn_link_list_t link_list;
    /* used for optimze strided slice to tensor view */
    struct
    {
        vx_node            cp_node;
        vx_tensor          src_tensor;
        vx_tensor          dst_tensor;
    };

    struct
    {
        int32_t *begin_dims;
        int32_t *end_dims;
        int32_t *stride_dims;
        int32_t begin_mask;
        int32_t end_mask;
        int32_t shrink_axis_mask;
    };

    vsi_bool is_dataconvert_op;
    vsi_bool is_optimized;

    strided_slice_param params;
} vsi_nn_strided_slice_lcl_data2;

typedef struct _vsi_nn_strided_slice_lcl_data_t
{
    vsi_nn_tensor_t *begin_dims_tensor;
    vsi_nn_tensor_t *end_dims_tensor;
    vsi_nn_tensor_t *stride_dims_tensor;
} vsi_nn_strided_slice_lcl_data_t;

typedef struct _vsi_nn_strided_slice_param
{
    /* local data must be the first. */
    vsi_nn_strided_slice_lcl_data_t local;

    const vx_int32 *begin_dims;
    vx_uint32 begin_dims_num;
    const vx_int32 *end_dims;
    vx_uint32 end_dims_num;
    const vx_int32 *stride_dims;
    vx_uint32 stride_dims_num;
    vx_int32 begin_mask;
    vx_int32 end_mask;
    vx_int32 shrink_axis_mask;
    int32_t new_axis_mask;

    vsi_nn_strided_slice_lcl_data2  * lcl2_data;
} vsi_nn_strided_slice_param;

#ifdef __cplusplus
}
#endif

#endif
