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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_test.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vx_int32 get_slice_axis_value(vx_int32 value, vx_uint32 dimension_size)
{
    vx_int32 axis_vlaue = 0;
    if (value < 0)
        axis_vlaue = value + dimension_size;
    else
        axis_vlaue = value;
    return axis_vlaue;
}

static vx_int32 get_slice_mask_start_value(vx_int32 stride, vx_uint32 dimension_size)
{
    vx_int32 start_vlaue = 0;
    if (stride > 0)
        start_vlaue = 0;
    else
        start_vlaue = dimension_size - 1;
    return start_vlaue;
}

static vx_int32 get_slice_mask_stop_value(vx_int32 stride, vx_uint32 dimension_size)
{
    vx_int32 stop_vlaue = 0;
    if (stride > 0)
        stop_vlaue = dimension_size;
    else
        stop_vlaue = -1;
    return stop_vlaue;
}

static vx_int32 get_slice_clamp_stop(vx_int32 stride, vx_int32 stop, vx_uint32 dimension_size)
{
    vx_int32 stop_vlaue = 0;
    if (stride > 0)
    {
        stop_vlaue = vsi_nn_clamp(stop, 0, (vx_int32)dimension_size);
    }
    else
    {
        stop_vlaue = vsi_nn_clamp(stop, -1, (vx_int32)dimension_size - 1);
    }
    return stop_vlaue;
}

static vsi_bool _check_neg_start_end_dims
    (
    int32_t *start,
    int32_t *stop,
    uint32_t dims
    )
{
    uint32_t i = 0;

    for (i = 0; i < dims; i++)
    {
        if (start[i] < 0 || stop[i] < 0)
            return TRUE;
    }

    return FALSE;
} /* _is_same_quant */

static vsi_bool _get_stride_slice_start_stop_stride
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t i = 0;
    int32_t int32_value = 0;
    vsi_nn_strided_slice_param *p = &(self->nn_param.strided_slice);
    int32_t *start = p->lcl2_data->begin_dims;
    int32_t *stop = p->lcl2_data->end_dims;
    int32_t *stride = p->lcl2_data->stride_dims;
    strided_slice_param* params = &p->lcl2_data->params;

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i ++)
    {
        start[i]    = 0;
        stop[i]     = 1;
        stride[i]   = 1;
    }

    for (i = 0; i < params->stride_dims_num; ++i)
    {
        stride[i] = params->stride_dims[i];
    }

    for (i = 0; i < params->begin_dims_num; ++i)
    {
        int32_value = params->begin_dims[i];

        start[i] = get_slice_axis_value(int32_value, (uint32_t)inputs[0]->attr.size[i]);
    }

    for (i = 0; i < params->end_dims_num; ++i)
    {
        int32_value = params->end_dims[i];

        stop[i] = get_slice_axis_value(int32_value, (uint32_t)inputs[0]->attr.size[i]);
    }

    /*if the ith bit of mask is set, the start or stop will be the fullest possible range in that dimension.*/
    for (i = 0; i < (int32_t)inputs[0]->attr.dim_num; i ++)
    {
        if (params->begin_mask & (1 << i))
        {
            start[i] = get_slice_mask_start_value(stride[i], (uint32_t)inputs[0]->attr.size[i]);
        }

        start[i] = vsi_nn_clamp(start[i], 0, (vx_int32)(inputs[0]->attr.size[i] - 1));

        if (params->end_mask & (1 << i))
        {
            stop[i] = (int32_t)get_slice_mask_stop_value(stride[i], (uint32_t)inputs[0]->attr.size[i]);
        }

        stop[i] = (int32_t)get_slice_clamp_stop(stride[i], stop[i], (uint32_t)inputs[0]->attr.size[i]);

        if (params->shrink_axis_mask & (1 << i))
        {
            stop[i] = start[i] + 1;
        }
    }

    /* reset start stop and stride when output size is 1*/
    for (i = 0; i < (int32_t)outputs[0]->attr.dim_num; i ++)
    {
        if (outputs[0]->attr.size[i] == 1 && stride[i] < 0)
        {
            stride[i] = 1;
            stop[i] = start[i] + 1;
        }
    }

    if (_check_neg_start_end_dims(start, stop, inputs[0]->attr.dim_num))
    {
        memcpy(start, params->begin_dims, sizeof(int32_t) * params->begin_dims_num);
        memcpy(stop, params->end_dims, sizeof(int32_t) * params->end_dims_num);
        memcpy(stride, params->stride_dims, sizeof(int32_t) * params->stride_dims_num);
        p->lcl2_data->begin_mask = params->begin_mask;
        p->lcl2_data->end_mask = params->end_mask;
        p->lcl2_data->shrink_axis_mask = params->shrink_axis_mask;
    }

    return TRUE;
}

static vsi_bool _check_is_same_shape(
    vsi_nn_tensor_t ** inputs,
    vsi_ssize_t *start,
    vsi_ssize_t *stop,
    vsi_ssize_t *stride
    )
{
    vsi_ssize_t i = 0;
    vsi_ssize_t dims = (vsi_ssize_t)inputs[0]->attr.dim_num;

    for (i = dims - 1; i >= 0; i --)
    {
        if (inputs[0]->attr.size[i] == 1)
        {
            dims --;
            continue;
        }
        else
            break;
    }

    for (i = 0; i < dims - 1; i++)
    {
        if (stride[i] != 1 || start[i] != 0 || stop[i] != (vsi_ssize_t)inputs[0]->attr.size[i])
            return FALSE;
    }

    if (stride[i] != 1)
        return FALSE;

    return TRUE;
}

static vsi_bool _is_same_quant
    (
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_dtype_t *src_dtype = NULL,*dst_dtype = NULL;

    src_dtype = &inputs[0]->attr.dtype;
    dst_dtype = &outputs[0]->attr.dtype;

    if(vsi_nn_DtypeCompare(src_dtype, dst_dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

static vsi_status copy_tensor_to_view
    (
    vsi_nn_node_t   * self,
    vx_tensor         src_tensor,
    vsi_nn_tensor_t * dst_in,
    vsi_size_t      * shape,
    vsi_bool          is_same_shape
    )
{
    vsi_status ret;
    vsi_nn_strided_slice_lcl_data2 * data = NULL;

    ret = VSI_SUCCESS;
    /* Malloc ptr */
    data = self->nn_param.strided_slice.lcl2_data;
    data->src_tensor = src_tensor;
    data->is_same_shape = is_same_shape;
    if (dst_in->t && !is_same_shape)
    {
        data->dst_tensor = vsi_nn_safe_reshape_tensor(dst_in->t, (void*)shape,
            (vsi_size_t)dst_in->attr.dim_num, sizeof(shape[0]));
    }
    else if (dst_in->t)
    {
        data->dst_tensor = dst_in->t;
    }

    data->is_dataconvert_op = TRUE;

    return ret;
} /* copy_tensor_to_view() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_nn_stride_slice_params_t param;
    vsi_nn_tensor_t *begin_dims_tensor = NULL;
    vsi_nn_tensor_t *end_dims_tensor = NULL;
    vsi_nn_tensor_t *stride_dims_tensor = NULL;
    vsi_nn_tensor_t *output_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    int32_t   *start_dims = NULL;
    int32_t   *stop_dims = NULL;
    int32_t   *stride_dims = NULL;
    vsi_nn_strided_slice_lcl_data2 * p = self->nn_param.strided_slice.lcl2_data;
    strided_slice_param* params = &p->params;

    start_dims = p->begin_dims;
    stop_dims = p->end_dims;
    stride_dims = p->stride_dims;

    if (TRUE == p->is_optimized)
    {
        vx_tensor dst_tensor = NULL;

        if (p->is_dataconvert_op)
        {
            dst_tensor = p->dst_tensor ? p->dst_tensor : outputs[0]->t;
            p->cp_node = vxTensorCopyNode(self->graph->g,
                    p->src_tensor, dst_tensor );
            if ( NULL == p->cp_node )
            {
                VSILOGE( "Create vxTensorCopyNode fail." );
                status = VSI_FAILURE;
            }
        }
    }
    else
    {
        vsi_size_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
        uint32_t dims = inputs[0]->attr.dim_num;
        int32_t shrink_axis_mask = params->shrink_axis_mask;
        int32_t new_axis_mask = params->new_axis_mask;
        int32_t num_add_axis = params->num_add_axis;

        memset(&param, 0, sizeof(vx_nn_stride_slice_params_t));

        memset(&attr, 0, sizeof(attr));
        attr.size[0] = params->begin_dims_num;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        begin_dims_tensor = vsi_nn_CreateTensorFromData(
            self->graph,
            (uint8_t *)start_dims,
            &attr);
        if ( NULL == begin_dims_tensor )
        {
            VSILOGE("Create begin_dims_tensor fail.(strided_slice)");
            return VSI_FAILURE;
        }

        self->nn_param.strided_slice.local.begin_dims_tensor = begin_dims_tensor;
        param.begin_dims = REQUIRED_IO(begin_dims_tensor);

        memset(&attr, 0, sizeof(attr));
        attr.size[0] = params->end_dims_num;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        end_dims_tensor = vsi_nn_CreateTensorFromData(
            self->graph,
            (uint8_t *)stop_dims,
            &attr);
        if ( NULL == end_dims_tensor )
        {
            VSILOGE("Create end_dims_tensor fail.(strided_slice)");
            return VSI_FAILURE;
        }

        self->nn_param.strided_slice.local.end_dims_tensor = end_dims_tensor;
        param.end_dims = REQUIRED_IO(end_dims_tensor);

        memset(&attr, 0, sizeof(attr));
        attr.size[0] = params->stride_dims_num;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        stride_dims_tensor = vsi_nn_CreateTensorFromData(
            self->graph,
            (uint8_t *)stride_dims,
            &attr);
        if ( NULL == stride_dims_tensor )
        {
            VSILOGE("Create stride_dims_tensor fail.(strided_slice)");
            return VSI_FAILURE;
        }

        self->nn_param.strided_slice.local.stride_dims_tensor = stride_dims_tensor;
        param.stride_dims = REQUIRED_IO(stride_dims_tensor);

        param.begin_mask = p->begin_mask;
        param.end_mask = p->end_mask;
        param.shrink_axis_mask = p->shrink_axis_mask;

        /* reshpae output tensor to keep output rank is the same as input's */
        memset(&sizes, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
        memcpy(&sizes, &outputs[0]->attr.size, sizeof(int32_t) * outputs[0]->attr.dim_num);

        if ((shrink_axis_mask && p->shrink_axis_mask == 0) ||
            new_axis_mask)
        {
            uint32_t i = 0;
            uint32_t j = 0, idx = 0;

            for (i = 0; i < inputs[0]->attr.dim_num + num_add_axis; i++)
            {
                if ( new_axis_mask & (1 << i) )
                {
                    j ++;
                    continue;
                }
                else if (shrink_axis_mask & (1 << i))
                {
                    sizes[idx ++] = 1;
                }
                else
                {
                    sizes[idx ++] = outputs[0]->attr.size[j ++];
                }
            }
        }

        output_tensor = vsi_nn_reshape_tensor(self->graph, outputs[0], sizes, dims);
        if ( NULL == output_tensor )
        {
            VSILOGE("Create output_tensor fail.(strided_slice)");
            return VSI_FAILURE;
        }

        self->n = vxTensorStrideSliceNode(
            self->graph->g,
            inputs[0]->t,
            &param,
            sizeof(vx_nn_stride_slice_params_t),
            output_tensor->t
            );

        if (output_tensor)
        {
            vsi_nn_ReleaseTensor(&output_tensor);
        }

        if ( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }
    }
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(STRIDED_SLICE, 1, 1)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_F16,          D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,     D_F16)
        IO_TYPE(D_I8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_SYM,     D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_F16)
        IO_TYPE(D_I16|Q_ASYM,   D_F16)
        IO_TYPE(D_I16|Q_SYM,    D_F16)
        IO_TYPE(D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_F32,          D_BF16)
        IO_TYPE(D_BF16,         D_F32)
        IO_TYPE(D_I32,          D_I32)
        IO_TYPE(D_BOOL8,        D_BOOL8)

        /* HW 9.0.1 */
        IO_TYPE(D_U8|Q_ASYM,    D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,    D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,    D_BF16)
        IO_TYPE(D_U8|Q_ASYM,    D_F32)
        IO_TYPE(D_U8|Q_ASYM,    D_I8|Q_DFP)

        IO_TYPE(D_I8|Q_DFP,     D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,     D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_BF16)
        IO_TYPE(D_I8|Q_DFP,     D_F32)

        IO_TYPE(D_I16|Q_DFP,    D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,    D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,    D_BF16)
        IO_TYPE(D_I16|Q_DFP,    D_F32)

        IO_TYPE(D_F16,          D_BF16)
        IO_TYPE(D_F16,          D_F32)
        IO_TYPE(D_U8|Q_ASYM,    D_I32|Q_ASYM)

        /* HW 9.1.1 */
        IO_TYPE(D_U4|Q_ASYM,    D_U4|Q_ASYM)
        IO_TYPE(D_U4|Q_SYM,     D_U4|Q_SYM)
        IO_TYPE(D_I4|Q_ASYM,    D_I4|Q_ASYM)
        IO_TYPE(D_I4|Q_SYM,     D_I4|Q_SYM)

    END_IO_TYPE_DECL(STRIDED_SLICE)
    if (!VALIDATE_OP_IO_TYPES(STRIDED_SLICE, self, inputs, 1, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

int32_t _reverse_mask_bits(int32_t mask, int32_t dims)
{
    int32_t i = 0;
    int32_t new_mask = 0;
    int32_t bits = mask;
    int32_t leading_one = 0;

    for (leading_one = 0; leading_one < VSI_NN_MAX_DIM_NUM; leading_one ++)
    {
        if ( bits == 0 )
        {
            break;
        }

        bits >>= 1;
    }

    dims = vsi_nn_max(dims, leading_one);
    for (i = 0; i < dims; i++)
    {
        int32_t offset = dims - i - 1;
        if (mask & (1 << i))
        {
            new_mask |= (1 << offset);
        }
    }

    return new_mask;
}

void _reverse_indices(int32_t *dst, const int32_t *src, int32_t dims)
{
    int32_t i = 0;

    for (i = 0; i < dims; i++)
    {
        dst[dims - i - 1] = src[i];
    }
}

static vsi_bool _build_strided_slice_params(vsi_nn_strided_slice_param * op_params, int32_t input_dims)
{
    uint32_t i = 0;
    int32_t num_add_axis = 0;
    int32_t added_ellipsis = 0;
    int32_t begin_mask = op_params->begin_mask;
    int32_t end_mask = op_params->end_mask;
    int32_t shrink_axis_mask = op_params->shrink_axis_mask;
    int32_t new_axis_mask = op_params->new_axis_mask;
    int32_t start_indices[2][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t stop_indices[2][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t strides[2][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t start_mask = 0;
    int32_t stop_mask = 0;
    int32_t shrink_mask = 0;
    int32_t output_dims = input_dims;
    const int32_t *begin_dims = op_params->begin_dims;
    const int32_t *end_dims = op_params->end_dims;
    const int32_t *stride_dims = op_params->stride_dims;
    strided_slice_param *params = &op_params->lcl2_data->params;

    begin_mask = _reverse_mask_bits(begin_mask, input_dims);
    end_mask   = _reverse_mask_bits(end_mask, input_dims);
    shrink_axis_mask = _reverse_mask_bits(shrink_axis_mask, input_dims);
    _reverse_indices(start_indices[0], begin_dims, op_params->begin_dims_num);
    _reverse_indices(stop_indices[0], end_dims, op_params->end_dims_num);
    _reverse_indices(strides[0], stride_dims, op_params->stride_dims_num);

    for (i = 0; i < op_params->begin_dims_num; i++)
    {
        if ( op_params->new_axis_mask & (1 << i))
        {
            num_add_axis ++;
            output_dims ++;
        }
    }

    for (i = 0; i < (uint32_t)(input_dims + num_add_axis); i++)
    {
        if ( op_params->shrink_axis_mask & (1 << i))
        {
            output_dims --;
        }
    }

    params->new_axis_mask = new_axis_mask;
    new_axis_mask = _reverse_mask_bits(new_axis_mask, output_dims);

    params->num_add_axis = num_add_axis;

    for (i = 0; i < (uint32_t)(input_dims + num_add_axis); i++)
    {
        if ( new_axis_mask & (1 << i) )
        {
            continue;
        }
        else if (i >= op_params->begin_dims_num + added_ellipsis)
        {
            start_mask |= (1 << params->begin_dims_num);
            stop_mask |= (1 << params->end_dims_num);
            start_indices[1][params->begin_dims_num ++ ] = 0;
            stop_indices[1][params->end_dims_num ++] = 0;
            strides[1][params->stride_dims_num ++] = 1;
        }
        else
        {
            int32_t orig_idx = i - added_ellipsis;

            if (begin_mask & (1 << orig_idx))
            {
                start_mask |= (1 << params->begin_dims_num);
            }

            if (end_mask & (1 << orig_idx))
            {
                stop_mask |= (1 << params->end_dims_num);
            }

            if (shrink_axis_mask & (1 << orig_idx))
            {
                shrink_mask |= (1 << params->begin_dims_num);
            }

            start_indices[1][params->begin_dims_num ++] = start_indices[0][orig_idx];
            stop_indices[1][params->end_dims_num ++] = stop_indices[0][orig_idx];
            strides[1][params->stride_dims_num ++] = strides[0][orig_idx];
        }
    }

    params->begin_mask = _reverse_mask_bits(start_mask, input_dims);
    params->end_mask   = _reverse_mask_bits(stop_mask, input_dims);
    params->shrink_axis_mask = _reverse_mask_bits(shrink_mask, input_dims);
    _reverse_indices(params->begin_dims, start_indices[1], params->begin_dims_num);
    _reverse_indices(params->end_dims, stop_indices[1], params->end_dims_num);
    _reverse_indices(params->stride_dims, strides[1], params->stride_dims_num);

    return TRUE;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i = 0;
    vsi_nn_strided_slice_param *p = &(self->nn_param.strided_slice);
    strided_slice_param *params = &p->lcl2_data->params;

    if ( vsi_nn_compareVersion(self->graph, 1, 1, 32) == -1
        && self->nn_param.strided_slice.begin_dims_num == 0)
    {
        self->nn_param.strided_slice.begin_dims_num = inputs[0]->attr.dim_num;
        self->nn_param.strided_slice.end_dims_num = inputs[0]->attr.dim_num;
        self->nn_param.strided_slice.stride_dims_num = inputs[0]->attr.dim_num;
    }

    _build_strided_slice_params(p, inputs[0]->attr.dim_num);

    /* TODO: Add code to comput outputs' shape. */

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        int32_t idx = 0;
        uint32_t shape[VSI_NN_MAX_DIM_NUM] = {0};

        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            vx_int32 begin = 0, end = 1, stride = 1;
            vx_int32 input_size = (int32_t)inputs[0]->attr.size[i];
            vx_int32 output_size = 0;
            vx_int32 j;

            begin = get_slice_axis_value(params->begin_dims[i], input_size);
            end = get_slice_axis_value(params->end_dims[i], input_size);
            stride = params->stride_dims[i];
            if (params->begin_mask & (1 << i))
            {
                begin = get_slice_mask_start_value(stride, input_size);
            }
            begin = vsi_nn_clamp(begin, 0, (vx_int32)(input_size - 1));
            if (params->shrink_axis_mask & (1 << i))
            {
                end = begin + 1;
            }

            if (params->end_mask & (1 << i))
            {
                end = get_slice_mask_stop_value(stride, input_size);
            }
            end = get_slice_clamp_stop(stride, end, input_size);
            for (j = begin; !((stride > 0) ? (j >= end) : (j <= end)); j += stride)
            {
                output_size++;
            }
            shape[i] = output_size;
        }
        outputs[0]->attr.dim_num = 0;
        for (idx = 0, i = 0; i < inputs[0]->attr.dim_num + params->num_add_axis; i++)
        {
            if (p->new_axis_mask & (1 << i))
            {
                outputs[0]->attr.size[outputs[0]->
                    attr.dim_num] = 1;

                outputs[0]->attr.dim_num++;
                continue;
            }
            else if (params->shrink_axis_mask & (1 << idx))
            {
                idx ++;
                continue;
            }

            outputs[0]->attr.size[outputs[0]->
                attr.dim_num] = shape[idx ++];

            outputs[0]->attr.dim_num++;
        }
    }

    _get_stride_slice_start_stop_stride(self, inputs, outputs);

    return TRUE;
} /* op_setup() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status     status = VSI_SUCCESS;
    int32_t        i = 0;
    vx_tensor      in_view_tensor = NULL;
    vsi_nn_strided_slice_param *p = &(self->nn_param.strided_slice);
    vsi_size_t     start[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t     end[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_ssize_t    start_dims[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_ssize_t    stop_dims[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_ssize_t    stride_dims[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t     shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool       is_same_quant_type = FALSE;
    vsi_bool       is_same_shape = TRUE;
    vsi_size_t     input_elements = 0;
    vsi_size_t     output_elements = 0;

    /* Only forward run stride_slice's optimize */
    if ( direction == VSI_NN_OPTIMIZE_BACKWARD )
    {
        return status;
    }

    for (i = 0; i< VSI_NN_MAX_DIM_NUM; i++)
    {
        start_dims[i] = p->lcl2_data->begin_dims[i];
        stop_dims[i] = p->lcl2_data->end_dims[i];
        stride_dims[i] = p->lcl2_data->stride_dims[i];

        shape[i] = (vsi_size_t)stop_dims[i] - (vsi_size_t)start_dims[i];
        if (shape[i] != outputs[0]->attr.size[i] &&
            i < (int32_t)outputs[0]->attr.dim_num)
        {
            is_same_shape = FALSE;
        }
    }

    if (_check_is_same_shape(inputs, start_dims, stop_dims, stride_dims) == FALSE)
        return status;

    VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);

    self->nn_param.strided_slice.lcl2_data->is_optimized = TRUE;

    is_same_quant_type = _is_same_quant(inputs, outputs);
    input_elements = vsi_nn_GetElementNum( inputs[0] );
    output_elements = vsi_nn_GetElementNum( outputs[0] );
    if (NULL != outputs[0]->t && NULL == inputs[0]->t &&
        is_same_quant_type && input_elements == output_elements)
    {
        inputs[0]->t = vsi_nn_safe_reshape_tensor( outputs[0]->t,
            (void*)inputs[0]->attr.size, (vsi_size_t)inputs[0]->attr.dim_num,
            sizeof(inputs[0]->attr.size[0]) );
    }
    else if (inputs[0]->attr.dim_num == outputs[0]->attr.dim_num)
    {
        if ( NULL == inputs[0]->t )
        {
            vsi_nn_TensorReinit( self->graph, inputs[0] );
        }
        /* Create tensor from view */
        memcpy( start, (vsi_size_t*)start_dims, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM );
        memcpy( end, (vsi_size_t*)stop_dims, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM );
        in_view_tensor = vsi_nn_CreateViewTensor(self->graph, start, end, inputs[0]);
        if ( NULL == in_view_tensor )
        {
            VSILOGE( "Create tensor %d from view fail.", i );
            status = VSI_FAILURE;
            goto OnError;
        }

        if ( NULL != outputs[0]->t || is_same_quant_type == FALSE)
        {
            VSILOGI( "stride slice copy tensor.");
            // Copy old tensor values to the new address.
            status = copy_tensor_to_view( self, in_view_tensor, outputs[0], shape, is_same_shape);
            if ( VSI_FAILURE == status )
            {
                goto OnError;
            }
        }
        else
        {
            outputs[0]->t = in_view_tensor;
        }
    }
    else
    {
        self->nn_param.strided_slice.lcl2_data->is_optimized = FALSE;
    }

OnError:
    return status;
} /* op_optimize() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_strided_slice_lcl_data2 * lcl2_data;
    strided_slice_param *params = NULL;
    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    lcl2_data = self->nn_param.strided_slice.lcl2_data;
    params = &lcl2_data->params;

    if (self->n)
    {
        if( NULL != self && NULL != self->n )
        {
            vxReleaseNode( &self->n );
            self->n = NULL;
        }
    }

    vsi_nn_safe_free( params->begin_dims );
    vsi_nn_safe_free( params->end_dims );
    vsi_nn_safe_free( params->stride_dims );

    if (lcl2_data && lcl2_data->cp_node)
    {
        vxReleaseNode( &lcl2_data->cp_node );
    }

    if (lcl2_data && lcl2_data->src_tensor)
    {
        vxReleaseTensor( &lcl2_data->src_tensor );
    }

    if (lcl2_data && lcl2_data->dst_tensor && !lcl2_data->is_same_shape)
    {
        vxReleaseTensor( &lcl2_data->dst_tensor );
    }

    if (lcl2_data && lcl2_data->begin_dims)
    {
        free(lcl2_data->begin_dims);
    }

    if (lcl2_data && lcl2_data->end_dims)
    {
        free(lcl2_data->end_dims);
    }

    if (lcl2_data && lcl2_data->stride_dims)
    {
        free(lcl2_data->stride_dims);
    }

    if (lcl2_data)
    {
        free( lcl2_data );
    }

    if (self->nn_param.strided_slice.local.begin_dims_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.strided_slice.local.begin_dims_tensor));
    }
    if (self->nn_param.strided_slice.local.end_dims_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.strided_slice.local.end_dims_tensor));
    }
    if (self->nn_param.strided_slice.local.stride_dims_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.strided_slice.local.stride_dims_tensor));
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_strided_slice_lcl_data2 * lcl2_data = NULL;
    strided_slice_param* params = NULL;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 32) == -1)
    {
        self->nn_param.strided_slice.new_axis_mask = 0;
    }

    self->nn_param.strided_slice.lcl2_data =
        (vsi_nn_strided_slice_lcl_data2 *)malloc(sizeof(vsi_nn_strided_slice_lcl_data2));
    if (NULL == self->nn_param.strided_slice.lcl2_data)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    lcl2_data = self->nn_param.strided_slice.lcl2_data;

    memset( lcl2_data, 0, sizeof(vsi_nn_strided_slice_lcl_data2) );

    params = &lcl2_data->params;

    lcl2_data->begin_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == lcl2_data->begin_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(lcl2_data->begin_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    params->begin_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == params->begin_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(params->begin_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    lcl2_data->end_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == lcl2_data->end_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(lcl2_data->end_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    params->end_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == params->end_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(params->end_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    lcl2_data->stride_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == lcl2_data->stride_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(lcl2_data->stride_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    params->stride_dims =
        (int32_t *)malloc(sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);
    if (NULL == params->stride_dims)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(params->stride_dims, 0, sizeof(int32_t) * VSI_NN_MAX_DIM_NUM);

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ STRIDED_SLICE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
