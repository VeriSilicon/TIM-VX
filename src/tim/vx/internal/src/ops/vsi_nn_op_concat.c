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
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_link_list.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

static int32_t _get_input_num
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs
    )
{
    int32_t num;
    num = (int32_t)(self->input.num - 1);
    while( num >= 0 && NULL == inputs[num] )
    {
        num --;
    }
    if( 0 > num )
    {
        return -1;
    }

    num++;
    return num;
}

static vsi_bool _is_same_quant
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i,num;
    vsi_nn_dtype_t *dtype,*_dtype;

    dtype = NULL;
    /* check inputs dtype */
    num = _get_input_num(self, inputs);
    for(i = 0; i < num; i++)
    {
        if(NULL == dtype)
        {
            dtype = &inputs[i]->attr.dtype;
            continue;
        }

        _dtype = &inputs[i]->attr.dtype;
        if(vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
        {
            return FALSE;
        }

        dtype = _dtype;
    }

    /* check outputs dtype */
    _dtype = &outputs[0]->attr.dtype;
    if(vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

static vsi_bool _is_highest_dimension
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    uint32_t axis = self->nn_param.concat.axis;
    uint32_t dim = outputs[0]->attr.dim_num;

    /*
        If the concat op need to be optimized to tensor view, the memory must be continues.
        1. axis is in the highest dimension
        2. the highest dimension is 1, and axis is in the second highest dimension
    */
    if(axis == dim - 1)
    {
        ret = TRUE;
    }
    if((outputs[0]->attr.size[dim - 1] == 1) && (axis == dim - 2))
    {
        ret = TRUE;
    }
    return ret;
} /* _is_highest_dimension() */

static vsi_status copy_tensor_to_view
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t * src_in,
    uint32_t         axis,
    vx_tensor         dst_tensor
    )
{
    vsi_status ret;
    vsi_nn_concat_lcl_data * data;

    ret = VSI_SUCCESS;
    /* Malloc ptr */
    data = (vsi_nn_concat_lcl_data *)malloc( sizeof(vsi_nn_concat_lcl_data) );
    if( NULL == data )
    {
        VSILOGE( "Create concat local data fail." );
        return VSI_FAILURE;
    }
    memset( data, 0, sizeof(vsi_nn_concat_lcl_data) );
    data->src_tensor = src_in->t;
    data->dst_tensor = dst_tensor;

    /* Store node, ptr */
    vsi_nn_LinkListPushStart(
        (vsi_nn_link_list_t **)&self->nn_param.concat.lcl_data,
        (vsi_nn_link_list_t *)data );

    return ret;
} /* copy_tensor_to_view() */

static vx_node _create_vx_concat
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vx_nn_concat_params_t param;
    vx_node node;
    int32_t num,i;
    vsi_nn_concat_lcl_data *data = NULL;
    vx_tensor *tensors = NULL;
    vsi_status status = VSI_FAILURE;
    vx_enum rank = VX_TENSOR_RANK_WHCN;

    num = _get_input_num(self, inputs);
    if(num < 0)
    {
        return NULL;
    }

    tensors = (vx_tensor *)malloc(sizeof(vx_tensor) * num);
    if(NULL == tensors)
    {
        return NULL;
    }

    node = NULL;
    for(i = 0; i < num; i++)
    {
        tensors[i] = inputs[i]->t;
        status = vxSetTensorAttribute(tensors[i], VX_TENSOR_RANK, &rank, sizeof(vx_enum));
        if(VSI_SUCCESS != status)
        {
            goto final;
        }
    }
    status = vxSetTensorAttribute(outputs[0]->t, VX_TENSOR_RANK, &rank, sizeof(vx_enum));
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    data = (vsi_nn_concat_lcl_data *)malloc(sizeof(vsi_nn_concat_lcl_data));
    if(NULL == data)
    {
        goto final;
    }

    memset(data, 0, sizeof(vsi_nn_concat_lcl_data));
    data->array = vxCreateTensorObjectArray(self->graph->ctx->c,
                                            num,
                                            &tensors[0]);
    if(NULL == data->array)
    {
        free(data);
        data = NULL;
        goto final;
    }
    param.axis = self->nn_param.concat.axis;
    self->nn_param.concat.lcl_data = data;

    node = vxConcatIndefiniteLayer(self->graph->g,
                                   data->array,
                                   &param,
                                   sizeof(param),
                                   outputs[0]->t);

final:
    if(tensors)
    {
        free(tensors);
    }
    return node;
} /* _create_vx_concat() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_concat_lcl_data * iter;

    status = VSI_SUCCESS;
    self->n = NULL;
    if(_is_highest_dimension(self, outputs) && _is_same_quant(self, inputs, outputs))
    {
        iter = self->nn_param.concat.lcl_data;
        while( NULL != iter )
        {
            iter->cp_node = vxTensorCopyNode(self->graph->g,
                iter->src_tensor, iter->dst_tensor );
            if( NULL == iter->cp_node )
            {
                VSILOGE( "Create vxTensorCopyNode fail." );
                status = VSI_FAILURE;
                break;
            }
            iter = (vsi_nn_concat_lcl_data *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter );
        }
    }
    else
    {
        self->n = _create_vx_concat(self, inputs, outputs);
        if(NULL == self->n)
        {
            status = VSI_FAILURE;
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
    vsi_bool ret;
    uint32_t axis,j;
    int32_t  num,i;

    ret = TRUE;
    axis = self->nn_param.concat.axis;
    num = _get_input_num(self, inputs);
    if(num < 0)
    {
        return FALSE;
    }
    for( i = 1; i < num; i ++ )
    {
        if( inputs[i]->attr.dim_num != inputs[i - 1]->attr.dim_num )
        {
            VSILOGE( "Concat input dims num(%d vs %d)",
                inputs[i]->attr.dim_num,
                inputs[i - 1]->attr.dim_num
                );
            ret = FALSE;
            break;
        }
        if( outputs[0]->attr.dim_num != VSI_NN_DIM_AUTO &&
            outputs[0]->attr.dim_num != inputs[i]->attr.dim_num )
        {
            VSILOGE( "Concat output dims num(%d vs %d)",
                outputs[0]->attr.dim_num,
                inputs[i]->attr.dim_num
                );
            ret = FALSE;
            break;
        }
        for( j = 0; j < inputs[i]->attr.dim_num; j ++ )
        {
            if( axis == j )
            {
                continue;
            }
            if( inputs[i]->attr.size[j] != inputs[i - 1]->attr.size[j] )
            {
                VSILOGE( "Concat input dims size(%d vs %d)",
                    inputs[i]->attr.size[j],
                    inputs[i - 1]->attr.size[j]
                );
                ret = FALSE;
                break;
            }
            if( outputs[0]->attr.dim_num != VSI_NN_DIM_AUTO &&
                outputs[0]->attr.size[j] != inputs[i]->attr.size[j])
            {
                VSILOGE( "Concat output dims size(%d vs %d)",
                    outputs[0]->attr.size[j],
                    inputs[i]->attr.size[j]
                );
                ret = FALSE;
                break;
            }
        }
        if( FALSE == ret )
        {
            break;
        }
    }

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t         num,i;
    vsi_bool        ret;
    uint32_t        axis;

    self->nn_param.concat.lcl_data = NULL;
    ret = TRUE;
    if( VSI_NN_DIM_AUTO != outputs[0]->attr.dim_num )
    {
        return ret;
    }

    num = _get_input_num(self, inputs);
    if(num < 0)
    {
        return FALSE;
    }
    axis = self->nn_param.concat.axis;
    memcpy( outputs[0]->attr.size, inputs[0]->attr.size,
        sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    for( i = 1; i < num; i ++ )
    {
        outputs[0]->attr.size[axis] += inputs[i]->attr.size[axis];
    }
    return ret;
} /* op_setup() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status     status;
    int32_t        num,i;
    uint32_t       axis;
    vx_tensor      in_view_tensor;
    uint32_t       start[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t       end[VSI_NN_MAX_DIM_NUM] = { 0 };

    status = VSI_SUCCESS;
    /* we don't create tensor view if the axis is not the highest dimension */
    if (_is_highest_dimension(self, outputs) == FALSE ||
        _is_same_quant(self, inputs, outputs) == FALSE)
    {
        return status;
    }
    /* Only backward run concat's optimize */
    if( direction == VSI_NN_OPTIMIZE_FORWARD )
    {
        return status;
    }

    VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
    num = _get_input_num(self, inputs);
    if(num < 0)
    {
        return VSI_FAILURE;
    }
    axis = self->nn_param.concat.axis;

    if( NULL == outputs[0]->t )
    {
        vsi_nn_TensorReinit( self->graph, outputs[0] );
    }

    /* Create tensor from view */
    memset( start, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
    memset( end, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
    end[0] = inputs[0]->attr.size[0];
    end[1] = inputs[0]->attr.size[1];
    end[2] = inputs[0]->attr.size[2];
    end[3] = inputs[0]->attr.size[3];
    end[axis] = 0;
    for( i = 0; i < num; i++ )
    {
        start[axis] = end[axis];
        end[axis] += inputs[i]->attr.size[axis];
        in_view_tensor = vsi_nn_CreateViewTensor(self->graph, start, end, outputs[0]);
        if( NULL == in_view_tensor )
        {
            VSILOGE( "Create a tensor view fail.");
            status = VSI_FAILURE;
            break;
        }

        if( NULL != inputs[i]->t )
        {
            VSILOGI( "Concat copy %d tensor.", i );
            // Copy old tensor values to the new address.
            status = copy_tensor_to_view( self, inputs[i], axis, in_view_tensor );
            if( VSI_FAILURE == status )
            {
                break;
            }
        }
        else
        {
            inputs[i]->t = in_view_tensor;
        }
    }

    return status;
} /* op_optimize() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_concat_lcl_data * data;
    vsi_nn_concat_lcl_data * tmp;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    data = self->nn_param.concat.lcl_data;
    if(self->n)
    {
        if( NULL != self && NULL != self->n )
        {
            if(data && data->array)
            {
                vxReleaseObjectArray(&data->array);
                free(data);
                data = NULL;
            }
            vxReleaseNode( &self->n );
            self->n = NULL;
        }
    }
    else
    {
        while( NULL != data )
        {
            tmp = (vsi_nn_concat_lcl_data *)vsi_nn_LinkListPopStart(
                (vsi_nn_link_list_t **)&data );
            vxReleaseNode( &tmp->cp_node );
            vxReleaseTensor( &tmp->dst_tensor );
            free( tmp );
        }
    }

    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
// TODO: Fix the concat input num.
DEF_OP_REG
    (
    /* op_name    */ CONCAT,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 16,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

