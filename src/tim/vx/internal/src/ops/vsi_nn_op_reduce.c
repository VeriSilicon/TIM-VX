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
#include "vsi_nn_platform.h"

#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#define _ARG_NUM            (6)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

typedef struct _vsi_nn_reduce_lcl2_data_t
{
    vsi_nn_tensor_t *reshaped_input;
    vsi_nn_tensor_t *reshaped_output;
    vsi_nn_tensor_t *reshaped_input1;
    vsi_nn_tensor_t *reshaped_output1;
    vsi_nn_tensor_t *reshaped_tmp;
    vsi_nn_tensor_t *axis_tensor2;
    int32_t axes[VSI_NN_MAX_DIM_NUM];
    int32_t axes_num;
} vsi_nn_reduce_lcl2_data_t;

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_REDUCE_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vx_uint32 i;
    vx_uint32 cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */

static vx_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    vx_uint32 num
    )
{
    vx_status status;
    vx_context ctx;
    vsi_nn_reduce_param * p = NULL;
    if( 0 == num )
    {
        return VX_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &node->nn_param.reduce;
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VX_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, axis_num );
    _SET_PARAM( 1, VX_TYPE_INT32, keep_dim );
    _SET_PARAM( 2, VX_TYPE_INT32, axis[0] );
    _SET_PARAM( 3, VX_TYPE_INT32, axis[1] );
    _SET_PARAM( 4, VX_TYPE_INT32, axis[2] );
    _SET_PARAM( 5, VX_TYPE_INT32, axis[3] );

#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    vx_uint32 num
    )
{
    vx_uint32 i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */

static vx_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vx_status status = VX_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args = NULL;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VX_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    NULL
};
#endif

static vsi_status op_comput_reduce_mean(vsi_nn_node_t * self,
                                        vsi_nn_tensor_t *axis_tensor,
                                        vx_bool keep_dim,
                                        vx_tensor input_t,
                                        vx_tensor output_t)
{
    vsi_status status = VSI_FAILURE;
    vx_nn_mean_params_t para;

    para.axis = REQUIRED_IO(axis_tensor);
    para.keep_dims = keep_dim;
    self->n = vxTensorMeanNode( self->graph->g, input_t, &para,
        sizeof(vx_nn_mean_params_t), output_t );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    return status;
}

static vsi_bool caculate_reshape_size(uint32_t* dim_value,
                                      uint32_t* re_sizes, uint32_t* re_sizes2,
                                      vx_int32 *resolved_dim, vx_int32 resolved_dim_count)
{
#define VSI_NN_MAX_IMAGE_WIDTH  (65536)
    vsi_bool enable_reshape = TRUE;
    uint32_t size_count = 1;
    uint32_t i = 0;
    uint32_t dim_num = *dim_value;
    if (dim_num > 4)
    {
        for (i = 4; i < dim_num; i++)
        {
            size_count *= re_sizes[i];
        }
    }

    if (re_sizes[0] * re_sizes[1] * re_sizes[2] < VSI_NN_MAX_IMAGE_WIDTH)
    {
        re_sizes2[0] = re_sizes[0] * re_sizes[1] * re_sizes[2];
        re_sizes2[1] = re_sizes[3];
        if (size_count != 1)
        {
            re_sizes2[2] = size_count;
            dim_num = 3;
        }
        else
        {
            dim_num = 2;
        }
        resolved_dim[resolved_dim_count - 1] = 1;
    }
    else if (re_sizes[0] * re_sizes[1] < VSI_NN_MAX_IMAGE_WIDTH)
    {
        re_sizes2[0] = re_sizes[0] * re_sizes[1];
        re_sizes2[1] = re_sizes[2];
        re_sizes2[2] = re_sizes[3];
        if (size_count != 1)
        {
            re_sizes2[3] = size_count;
            dim_num = 4;
        }
        else
        {
            dim_num = 3;
        }
        resolved_dim[resolved_dim_count - 1] = 2;
    }
    else if (re_sizes[1] * re_sizes[2] < VSI_NN_MAX_IMAGE_WIDTH)
    {
        re_sizes2[0] = re_sizes[0];
        re_sizes2[1] = re_sizes[1] * re_sizes[2];
        re_sizes2[2] = re_sizes[3];
        if (size_count != 1)
        {
            re_sizes2[3] = size_count;
            dim_num = 4;
        }
        else
        {
            dim_num = 3;
        }
        resolved_dim[resolved_dim_count - 1] = 2;
    }
    else
    {
        enable_reshape = FALSE;
    }
    *dim_value = dim_num;
#undef VSI_NN_MAX_IMAGE_WIDTH
    return enable_reshape;
}


static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
#if (USE_OVX_API == TRUE)

    if (self->nn_param.reduce.type == VSI_NN_REDUCE_MEAN)
    {
        vx_tensor input_t, output_t;
        vsi_nn_tensor_t *axis_tensor = NULL;
        vsi_nn_tensor_t *axis_tensor2 = NULL;
        vsi_nn_tensor_attr_t attr, attr2;
        vx_int32 resolved_dim[4]    = {-1, -1, -1, -1};
        vx_int32 resolved_dim_count = 0;
        uint32_t i = 0;
        uint32_t re_sizes[VSI_NN_MAX_DIM_NUM] = {1};
        uint32_t re_sizes2[VSI_NN_MAX_DIM_NUM] = {1};
        uint32_t dim_num;
        vsi_nn_tensor_t *mean_tmp_tensor = NULL;
        vsi_nn_tensor_t *reshaped_input1 = self->nn_param.reduce.local2->reshaped_input1;
        vsi_nn_tensor_t *reshaped_output1 = self->nn_param.reduce.local2->reshaped_output1;

        resolved_dim_count = self->nn_param.reduce.local2->axes_num;

        for (i = 0; i < (uint32_t)resolved_dim_count; i++)
        {
            resolved_dim[i] = self->nn_param.reduce.local2->axes[i];
        }

        for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
        {
            re_sizes[i]  = 1;
            re_sizes2[i] = 1;
        }
        memset(&attr2, 0, sizeof(attr));
        memcpy( &attr2, &reshaped_input1->attr, sizeof(vsi_nn_tensor_attr_t) );
        dim_num = reshaped_input1->attr.dim_num;
        for (i = 0; i < dim_num; i++)
        {
            attr2.size[i] = reshaped_input1->attr.size[i];
            re_sizes[i]  = reshaped_input1->attr.size[i];
        }
        if ((VSI_NN_TYPE_FLOAT32 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
            || (VSI_NN_TYPE_INT32 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
            || (VSI_NN_TYPE_UINT32 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
            || (VSI_NN_TYPE_UINT64 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
            )
        {
            attr2.dtype.vx_type  = VSI_NN_TYPE_FLOAT32;
        }
        else if (VSI_NN_TYPE_FLOAT64 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
        {
            attr2.dtype.vx_type  = VSI_NN_TYPE_FLOAT64;
        }
        else
        {
            attr2.dtype.vx_type  = VSI_NN_TYPE_FLOAT16;
        }

        attr2.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;

        if ((2 == resolved_dim_count && resolved_dim[0] < 3 && resolved_dim[1] < 3)
           || (1 == resolved_dim_count && resolved_dim[0] < 3)
           || (resolved_dim[resolved_dim_count - 1] > 3)
           || resolved_dim_count > 3)
        {
            memset(&attr, 0, sizeof(attr));
            attr.size[0] = resolved_dim_count;
            attr.dim_num = 1;
            attr.is_const = TRUE;
            attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            axis_tensor = vsi_nn_CreateTensorFromData(
                self->graph,
                (uint8_t *)resolved_dim,
                &attr);
            if( NULL == axis_tensor )
            {
                VSILOGE("Create axis_tensor fail.(reduce)");
                return VSI_FAILURE;
            }

            self->nn_param.reduce.local.axis_tensor = axis_tensor;
            input_t  = reshaped_input1->t;
            output_t = reshaped_output1->t;
            status = op_comput_reduce_mean(self,
                                           axis_tensor,
                                           self->nn_param.reduce.keep_dim,
                                           input_t,
                                           output_t);
        }
        else if (3 == resolved_dim[resolved_dim_count - 1] && resolved_dim_count < 3)
        {
            if (1 == resolved_dim_count)
            {
                memset(&attr, 0, sizeof(attr));
                attr.size[0] = resolved_dim_count;
                attr.dim_num = 1;
                attr.is_const = TRUE;
                attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
                attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
                axis_tensor = vsi_nn_CreateTensorFromData(
                    self->graph,
                    (uint8_t *)resolved_dim,
                    &attr);
                if( NULL == axis_tensor )
                {
                    VSILOGE("Create axis_tensor fail.(reduce)");
                    return VSI_FAILURE;
                }

                self->nn_param.reduce.local.axis_tensor = axis_tensor;
                input_t  = reshaped_input1->t;
                output_t = reshaped_output1->t;
                status = op_comput_reduce_mean(self,
                                               axis_tensor,
                                               self->nn_param.reduce.keep_dim,
                                               input_t,
                                               output_t);
            }
            else if (2 == resolved_dim_count)
            {
                vsi_bool enable_reshape = TRUE;

                attr2.size[resolved_dim[0]] = 1;
                attr2.vtl = FALSE;
                mean_tmp_tensor = vsi_nn_CreateTensor(self->graph, &attr2);
                self->nn_param.reduce.local2->reshaped_tmp = mean_tmp_tensor;
                re_sizes[resolved_dim[0]] = 1;
                memset(&attr, 0, sizeof(attr));
                attr.size[0] = 1;
                attr.dim_num = 1;
                attr.is_const = TRUE;
                attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
                attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
                axis_tensor = vsi_nn_CreateTensorFromData(
                    self->graph,
                    (uint8_t *)&resolved_dim[0],
                    &attr);
                if( NULL == axis_tensor )
                {
                    VSILOGE("Create axis_tensor fail.(reduce)");
                    return VSI_FAILURE;
                }
                self->nn_param.reduce.local.axis_tensor = axis_tensor;
                status = op_comput_reduce_mean(self,
                                               axis_tensor,
                                               self->nn_param.reduce.keep_dim,
                                               reshaped_input1->t,
                                               mean_tmp_tensor->t);

                enable_reshape = caculate_reshape_size(&dim_num, re_sizes, re_sizes2,
                                      resolved_dim, resolved_dim_count);

                if (enable_reshape)
                {
                    self->nn_param.reduce.local2->reshaped_input  =
                    vsi_nn_reshape_tensor(self->graph, mean_tmp_tensor, re_sizes2, dim_num);
                    re_sizes2[resolved_dim[resolved_dim_count - 1]] = 1;
                    self->nn_param.reduce.local2->reshaped_output =
                    vsi_nn_reshape_tensor(self->graph, reshaped_output1, re_sizes2, dim_num);
                }

                memset(&attr, 0, sizeof(attr));
                attr.size[0] = 1;
                attr.dim_num = 1;
                attr.is_const = TRUE;
                attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
                attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
                axis_tensor2 = vsi_nn_CreateTensorFromData(
                    self->graph,
                    (uint8_t *)&resolved_dim[1],
                    &attr);
                if( NULL == axis_tensor2 )
                {
                    VSILOGE("Create axis_tensor fail.(reduce)");
                    return VSI_FAILURE;
                }

                self->nn_param.reduce.local2->axis_tensor2 = axis_tensor2;
                if (self->nn_param.reduce.local2->reshaped_input)
                {
                    input_t  = self->nn_param.reduce.local2->reshaped_input->t;
                }
                else
                {
                    input_t  = mean_tmp_tensor->t;
                }
                if (self->nn_param.reduce.local2->reshaped_output)
                {
                    output_t = self->nn_param.reduce.local2->reshaped_output->t;
                }
                else
                {
                    output_t = reshaped_output1->t;
                }
                status = op_comput_reduce_mean(self,
                                               axis_tensor2,
                                               self->nn_param.reduce.keep_dim,
                                               input_t,
                                               output_t);
            }
        }
        else if (3 == resolved_dim_count)
        {
            vsi_bool enable_reshape = TRUE;

            attr2.size[resolved_dim[0]] = 1;
            attr2.size[resolved_dim[1]] = 1;
            attr2.vtl = FALSE;
            mean_tmp_tensor = vsi_nn_CreateTensor(self->graph, &attr2);
            self->nn_param.reduce.local2->reshaped_tmp = mean_tmp_tensor;
            re_sizes[resolved_dim[0]] = 1;
            re_sizes[resolved_dim[1]] = 1;
            memset(&attr, 0, sizeof(attr));
            attr.size[0] = 2;
            attr.dim_num = 1;
            attr.is_const = TRUE;
            attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            axis_tensor = vsi_nn_CreateTensorFromData(
                self->graph,
                (uint8_t *)&resolved_dim[0],
                &attr);
            if( NULL == axis_tensor )
            {
                VSILOGE("Create axis_tensor fail.(reduce)");
                return VSI_FAILURE;
            }
            self->nn_param.reduce.local.axis_tensor = axis_tensor;
            status = op_comput_reduce_mean(self,
                                            axis_tensor,
                                            self->nn_param.reduce.keep_dim,
                                            reshaped_input1->t,
                                            mean_tmp_tensor->t);
            if (3 == resolved_dim[resolved_dim_count - 1])
            {
                enable_reshape = caculate_reshape_size(&dim_num, re_sizes, re_sizes2,
                                      resolved_dim, resolved_dim_count);
                if (enable_reshape)
                {
                    self->nn_param.reduce.local2->reshaped_input  =
                    vsi_nn_reshape_tensor(self->graph, mean_tmp_tensor, re_sizes2, dim_num);
                    re_sizes2[resolved_dim[resolved_dim_count - 1]] = 1;
                    self->nn_param.reduce.local2->reshaped_output =
                    vsi_nn_reshape_tensor(self->graph, reshaped_output1, re_sizes2, dim_num);
                }
            }

            memset(&attr, 0, sizeof(attr));
            attr.size[0] = 1;
            attr.dim_num = 1;
            attr.is_const = TRUE;
            attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            axis_tensor2 = vsi_nn_CreateTensorFromData(
                self->graph,
                (uint8_t *)&resolved_dim[2],
                &attr);
            if( NULL == axis_tensor2 )
            {
                VSILOGE("Create axis_tensor fail.(reduce)");
                return VSI_FAILURE;
            }

            self->nn_param.reduce.local2->axis_tensor2 = axis_tensor2;
            if (self->nn_param.reduce.local2->reshaped_input)
            {
                input_t  = self->nn_param.reduce.local2->reshaped_input->t;
            }
            else
            {
                input_t  = mean_tmp_tensor->t;
            }
            if (self->nn_param.reduce.local2->reshaped_output)
            {
                output_t = self->nn_param.reduce.local2->reshaped_output->t;
            }
            else
            {
                output_t = reshaped_output1->t;
            }
            status = op_comput_reduce_mean(self,
                                            axis_tensor2,
                                            self->nn_param.reduce.keep_dim,
                                            input_t,
                                            output_t);
        }

    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_SUM ||
             self->nn_param.reduce.type == VSI_NN_REDUCE_MAX ||
             self->nn_param.reduce.type == VSI_NN_REDUCE_MIN ||
             self->nn_param.reduce.type == VSI_NN_REDUCE_ALL ||
             self->nn_param.reduce.type == VSI_NN_REDUCE_ANY ||
             self->nn_param.reduce.type == VSI_NN_REDUCE_PROD)
    {
        status = vsi_nn_internal_compute_node( self );
    }

#else
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_reduce";
    kernel_info.type = VX_KERNEL_TYPE_CPU;
    kernel_info.kernel = vx_kernel_REDUCE_list;
    kernel_info.kernel_index = 0;
    kernel_info.init_index = 0;

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
    if (kernel_info.resource_name) free(kernel_info.resource_name);
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
    }
#endif
    return status;
} /* op_compute() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if (self->nn_param.reduce.type == VSI_NN_REDUCE_SUM ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_MAX ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_MIN ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_ALL ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_ANY ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_PROD)
    {
        return vsi_nn_internal_optimize_node(self, direction );
    }
    else
    {
        return VSI_SUCCESS;
    }
} /* op_optimize() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static void op_set_reduce_param_value(vsi_nn_nn_param_t *nn_param,
                                    vsi_nn_op_t  type_name,
                                    vx_int32   *axis,
                                    vx_uint32   axis_num,
                                    vx_bool     keep_dim
                                    )
{
    if (type_name == VSI_NN_OP_REDUCESUM_INTERNAL)
    {
        nn_param->reducesum_internal.axis = axis;
        nn_param->reducesum_internal.axis_num = axis_num;
        nn_param->reducesum_internal.keep_dim = keep_dim;
    }
    else if (type_name == VSI_NN_OP_REDUCEMAX_INTERNAL)
    {
        nn_param->reducemax_internal.axis = axis;
        nn_param->reducemax_internal.axis_num = axis_num;
        nn_param->reducemax_internal.keep_dim = keep_dim;
    }
    else if (type_name == VSI_NN_OP_REDUCEMIN_INTERNAL)
    {
        nn_param->reducemin_internal.axis = axis;
        nn_param->reducemin_internal.axis_num = axis_num;
        nn_param->reducemin_internal.keep_dim = keep_dim;
    }
    else if (type_name == VSI_NN_OP_REDUCEPROD_INTERNAL)
    {
        nn_param->reduceprod_internal.axis = axis;
        nn_param->reduceprod_internal.axis_num = axis_num;
        nn_param->reduceprod_internal.keep_dim = keep_dim;
    }
    else if (type_name == VSI_NN_OP_REDUCEALL_INTERNAL)
    {
        nn_param->reduceall_internal.axis = axis;
        nn_param->reduceall_internal.axis_num = axis_num;
        nn_param->reduceall_internal.keep_dim = keep_dim;
    }
    else if (type_name == VSI_NN_OP_REDUCEANY_INTERNAL)
    {
        nn_param->reduceany_internal.axis = axis;
        nn_param->reduceany_internal.axis_num = axis_num;
        nn_param->reduceany_internal.keep_dim = keep_dim;
    }
}

static vsi_bool optimzation_input_size(
    const int32_t* shape_x, const size_t rank_x,
          int32_t* out_shape_x, int32_t* out_rank_x,
    const int32_t* resolved_dim, const int32_t resolved_dim_count,
          int32_t* resolved_dim_out,  int32_t* resolved_dim_out_count
    )
{
    int32_t i, j, k, out_i;
    vx_bool is_change = vx_false_e;
    int32_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    int32_t shape_out[VSI_NN_MAX_DIM_NUM] = { 0 };
    int32_t rank_out;
    int32_t dim_out;

    out_i = 0;
    for (i = 0; i < resolved_dim[0]; i++)
    {
        out_shape_x[out_i++] = shape_x[i];
    }

    j = 0;
    dim_out = 0;
    for (i = 0; i < (resolved_dim_count - 1); i++)
    {
       if ((resolved_dim[i] + 1) == resolved_dim[i + 1])
       {
            if (is_change)
            {
                shape[j++]  = shape_x[resolved_dim[i + 1]];
            }
            else
            {
                shape[j++]  = shape_x[resolved_dim[i]];
                shape[j++]  = shape_x[resolved_dim[i + 1]];
                is_change = vx_true_e;
            }
       }
       else
       {
            if (is_change)
            {
                vsi_nn_kernel_optimize_element_shape(
                        shape, (size_t)j,
                        shape_out, &rank_out );
                if (2 == rank_out &&  1 == shape_out[1])
                {
                    rank_out--;
                }
                for (k = 0; k < rank_out; k++)
                {
                    resolved_dim_out[dim_out++] = out_i;
                    out_shape_x[out_i++] = shape_out[k];
                }
                j = 0;
                is_change = vx_false_e;
            }
            else
            {
                  resolved_dim_out[dim_out++] = out_i;
                  out_shape_x[out_i++] = shape_x[resolved_dim[i]];
            }

            for ( k = resolved_dim[i] + 1; k < resolved_dim[i + 1]; k++ )
            {
                out_shape_x[out_i++] = shape_x[k];
            }
       }
    }

    if (is_change)
    {
         vsi_nn_kernel_optimize_element_shape(
                shape, (size_t)j,
                shape_out, &rank_out );
        if (2 == rank_out &&  1 == shape_out[1])
        {
            rank_out--;
        }
        for (k = 0; k < rank_out; k++)
        {
            resolved_dim_out[dim_out++] = out_i;
            out_shape_x[out_i++] = shape_out[k];
        }
    }
    else
    {
        resolved_dim_out[dim_out++] = out_i;
        out_shape_x[out_i++] = shape_x[resolved_dim[resolved_dim_count - 1]];
    }

    for (i = resolved_dim[resolved_dim_count - 1] + 1; i < (int32_t)rank_x; i++)
    {
        out_shape_x[out_i++] = shape_x[i];
    }

    if (1 == out_i)
    {
        out_shape_x[out_i++] = 1;
    }

    *out_rank_x = out_i;
    *resolved_dim_out_count = dim_out;

    return TRUE;
}

static vsi_bool op_set_reduce_axis(
                vsi_nn_node_t * self,
                vsi_nn_tensor_t ** inputs,
                int32_t* out_shape_x, int32_t* out_rank_x
                )
{
    uint32_t i = 0, j = 0;
    int32_t resolved_dim[4]    = {-1, -1, -1, -1};
    int32_t resolved_dim2[4]    = {-1, -1, -1, -1};
    int32_t resolved_dim_count = 0;
    int32_t resolved_dim_count2 = 0;
    vsi_bool is_loop = TRUE;

    for (i = 0; i < self->nn_param.reduce.axis_num; i++)
    {
        vx_int32 current_axis = self->nn_param.reduce.axis[i] < 0 ? \
        inputs[0]->attr.dim_num + self->nn_param.reduce.axis[i] : self->nn_param.reduce.axis[i];

        if (current_axis < 0 || current_axis >= (vx_int32)inputs[0]->attr.dim_num)
        {
            VSILOGE("error: the axis value must be in the range [0, %d)\n", inputs[0]->attr.dim_num);
            return FALSE;
        }

        for (j = 0; j < 4; j++)
        {
            if (resolved_dim[j] == current_axis)
                break;
        }

        if (j == 4)
            resolved_dim[resolved_dim_count++] = current_axis;
    }

    for (i = resolved_dim_count; is_loop && (i > 0); i--)
    {
        is_loop = FALSE;
        for (j = 1; j < i; j++)
        {
            if (resolved_dim[j] < resolved_dim[j - 1])
            {
                vx_int32 temp = 0;
                temp = resolved_dim[j];
                resolved_dim[j] = resolved_dim[j - 1];
                resolved_dim[j - 1] = temp;
                is_loop = TRUE;
            }
        }
    }

    if (resolved_dim_count > 1)
    {
        j = 0;
        for (i = 0; i < (uint32_t)resolved_dim_count; i++)
        {
            if (inputs[POST_PROCESS_OUTPUT]->attr.size[resolved_dim[i]] > 1)
            {
                resolved_dim[j] = resolved_dim[i];
                j++;
            }
        }
        if (j == 0)
        {
            j = 1;
        }
        resolved_dim_count = j;
    }

    if (( 1 == resolved_dim_count ))
    {
        resolved_dim2[0]    = resolved_dim[0];
        resolved_dim_count2 = resolved_dim_count;
        for (i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            out_shape_x[i] = (int32_t)(inputs[0]->attr.size[i]);
        }
        *out_rank_x = inputs[0]->attr.dim_num;
    }
    else
    {
        optimzation_input_size(
            (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
            out_shape_x, out_rank_x, resolved_dim, resolved_dim_count,
            resolved_dim2,  &resolved_dim_count2 );
    }


    for (i = 0; i < (uint32_t)resolved_dim_count2; i++)
    {
        self->nn_param.reduce.local2->axes[i] = resolved_dim2[i];
    }
    self->nn_param.reduce.local2->axes_num = resolved_dim_count2;

    return TRUE;
}


static vsi_bool op_set_reduce_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_op_t  type_name
    )
{
    uint32_t i = 0;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tmp_output_tensor[2] = {NULL, NULL};
    vsi_bool use_virtual_tensor = TRUE;
    uint32_t re_sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t re_sizes2[VSI_NN_MAX_DIM_NUM] = {1};
    vsi_nn_tensor_t* new_output = NULL;
    uint32_t dim_num;
    vx_int32 resolved_dim_count = 0;
    int32_t * axes = self->nn_param.reduce.local2->axes;
    vx_bool  is_use_float = vx_false_e;
    resolved_dim_count = self->nn_param.reduce.local2->axes_num;

    if ((VSI_NN_OP_REDUCESUM_INTERNAL == type_name) || (VSI_NN_OP_REDUCEPROD_INTERNAL == type_name))
    {
        is_use_float = vx_true_e;
    }

    vsi_nn_internal_init_node_wksp( self );

    memcpy( &attr, &inputs[POST_PROCESS_INPUT]->attr, sizeof(vsi_nn_tensor_attr_t) );
    dim_num = inputs[POST_PROCESS_INPUT]->attr.dim_num;

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        re_sizes[i]  = 1;
        re_sizes2[i] = 1;
    }

    for (i = 0; i < dim_num; i++)
    {
        attr.size[i] = inputs[POST_PROCESS_OUTPUT]->attr.size[i];
        re_sizes[i]  = inputs[POST_PROCESS_OUTPUT]->attr.size[i];
    }

    if (is_use_float)
    {
        if (  (VSI_NN_TYPE_FLOAT32 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
           || (VSI_NN_TYPE_INT32   == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
           || (VSI_NN_TYPE_UINT32  == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
           || (VSI_NN_TYPE_UINT64  == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
           )
        {
            attr.dtype.vx_type  = VSI_NN_TYPE_FLOAT32;
        }
        else if (VSI_NN_TYPE_FLOAT64 == inputs[POST_PROCESS_INPUT]->attr.dtype.vx_type)
        {
            attr.dtype.vx_type  = VSI_NN_TYPE_FLOAT64;
        }
        else
        {
            attr.dtype.vx_type  = VSI_NN_TYPE_FLOAT16;
        }
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    }

    if (1 == resolved_dim_count)
    {
        if (3 == axes[resolved_dim_count - 1])
        {
            vsi_bool enable_reshape = TRUE;
            enable_reshape = caculate_reshape_size(&dim_num, re_sizes, re_sizes2,
                                    axes, resolved_dim_count);
            if (enable_reshape)
            {
                self->nn_param.reduce.local2->reshaped_input  =
                vsi_nn_reshape_tensor(self->graph, inputs[0], re_sizes2, dim_num);
                re_sizes2[axes[resolved_dim_count - 1]] = 1;
                self->nn_param.reduce.local2->reshaped_output =
                vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes2, dim_num);
            }
        }

        curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
        op_set_reduce_param_value(&(curr->node->nn_param), type_name,
        axes, 1, self->nn_param.reduce.keep_dim);
        if (self->nn_param.reduce.local2->reshaped_input)
        {
            curr->inputs[0] = self->nn_param.reduce.local2->reshaped_input;
        }
        else
        {
            curr->inputs[0]  = inputs[0];
        }
        if (self->nn_param.reduce.local2->reshaped_output)
        {
            curr->outputs[0] = self->nn_param.reduce.local2->reshaped_output;
        }
        else
        {
            curr->outputs[0] = outputs[0];
        }
        vsi_nn_internal_setup_node(self, curr);
    }
    else if (2 == resolved_dim_count)
    {
        attr.size[axes[0]] = 1;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        tmp_output_tensor[0] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        re_sizes[axes[0]] = 1;

        curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
        op_set_reduce_param_value(&(curr->node->nn_param), type_name,
        &(axes[0]), 1, vx_true_e);
        curr->inputs[0]  = inputs[POST_PROCESS_INPUT];
        curr->outputs[0] = tmp_output_tensor[0]->t;
        vsi_nn_internal_setup_node( self, curr );

        if (3 == axes[resolved_dim_count - 1])
        {
            vsi_bool enable_reshape = TRUE;
            enable_reshape = caculate_reshape_size(&dim_num, re_sizes, re_sizes2,
                                    axes, resolved_dim_count);

            if (enable_reshape)
            {
                self->nn_param.reduce.local2->reshaped_input  =
                vsi_nn_reshape_tensor(self->graph, tmp_output_tensor[0]->t, re_sizes2, dim_num);
                re_sizes2[axes[resolved_dim_count - 1]] = 1;
                new_output = vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes2, dim_num);
            }
            else
            {
                re_sizes[axes[1]] = 1;
                new_output = vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes, dim_num);
            }
        }
        else
        {
            re_sizes[axes[1]] = 1;
            new_output = vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes, dim_num);
        }

        curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
        op_set_reduce_param_value(&(curr->node->nn_param), type_name,
        &(axes[1]), 1, vx_true_e);
        if (self->nn_param.reduce.local2->reshaped_input)
        {
            curr->inputs[0] = self->nn_param.reduce.local2->reshaped_input;
        }
        else
        {
            curr->inputs[0]  = tmp_output_tensor[0]->t;
        }
        curr->outputs[0] = new_output;
        self->nn_param.reduce.local2->reshaped_output = new_output;
        vsi_nn_internal_setup_node(self, curr);
    }
    else if (3 == resolved_dim_count)
    {
        attr.size[axes[0]] = 1;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        tmp_output_tensor[0] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        attr.size[axes[1]] = 1;
        tmp_output_tensor[1] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        re_sizes[axes[0]] = 1;
        re_sizes[axes[1]] = 1;

        curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
        op_set_reduce_param_value(&(curr->node->nn_param), type_name,
        &(axes[0]), 1, vx_true_e);
        curr->inputs[0]  = inputs[POST_PROCESS_INPUT];
        curr->outputs[0] = tmp_output_tensor[0]->t;
        vsi_nn_internal_setup_node( self, curr );

        curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
        op_set_reduce_param_value(&(curr->node->nn_param), type_name,
        &(axes[1]), 1, vx_true_e);
        curr->inputs[0]  = tmp_output_tensor[0]->t;
        curr->outputs[0] = tmp_output_tensor[1]->t;
        vsi_nn_internal_setup_node( self, curr );


        if (3 == axes[resolved_dim_count - 1])
        {
            vsi_bool enable_reshape = TRUE;
            enable_reshape = caculate_reshape_size(&dim_num, re_sizes, re_sizes2,
                                    axes, resolved_dim_count);
            if (enable_reshape)
            {
                self->nn_param.reduce.local2->reshaped_input  =
                vsi_nn_reshape_tensor(self->graph, tmp_output_tensor[1]->t, re_sizes2, dim_num);
                re_sizes2[axes[resolved_dim_count - 1]] = 1;
                new_output = vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes2, dim_num);
            }
            else
            {
                re_sizes[axes[2]] = 1;
                new_output = vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes, dim_num);
            }
        }
        else
        {
            re_sizes[axes[2]] = 1;
            new_output = vsi_nn_reshape_tensor(self->graph, outputs[0], re_sizes, dim_num);
        }

        curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
        op_set_reduce_param_value(&(curr->node->nn_param), type_name,
        &(axes[2]), 1, vx_true_e);
        if (self->nn_param.reduce.local2->reshaped_input)
        {
            curr->inputs[0]  = self->nn_param.reduce.local2->reshaped_input;
        }
        else
        {
            curr->inputs[0]  = tmp_output_tensor[1]->t;
        }
        curr->outputs[0] = new_output;
        self->nn_param.reduce.local2->reshaped_output = new_output;
        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        VSILOGE("error: resolved_dim_count is %d\n", resolved_dim_count);
        return FALSE;
    }
    return TRUE;
}


static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_tensor_t* reshape_in_t[1] = { NULL };
    vsi_nn_tensor_t* reshape_out_t[1] = { NULL };
    int32_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    int32_t new_rank = 0;
    int32_t j;
    if (self->nn_param.reduce.type != VSI_NN_REDUCE_MEAN &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_SUM  &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_MAX  &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_MIN  &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_ALL  &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_ANY  &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_PROD)
    {
        VSILOGE("The type of reduce is not supported now.(reduce)");
        return FALSE;
    }
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        int valid_dim_num = inputs[0]->attr.dim_num;
        uint32_t i;
        char dim_map[VSI_NN_MAX_DIM_NUM] = {0};

        for (i = 0; i < self->nn_param.reduce.axis_num; i++)
        {
            int index = self->nn_param.reduce.axis[i];
            if (dim_map[index] == 0) {
                dim_map[index] = 1;
                valid_dim_num --;
            }
        }

        if (self->nn_param.reduce.keep_dim)
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
            for (i = 0; i < inputs[0]->attr.dim_num; i++)
            {
                if (dim_map[i] == 0)
                {
                    outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
                }
                else
                {
                    outputs[0]->attr.size[i] = 1;
                }
            }
        }
        else
        {
            int index = 0;
            if (valid_dim_num == 0)
            {
                outputs[0]->attr.dim_num = 2;
                outputs[0]->attr.size[0] = 1;
                outputs[0]->attr.size[1] = 1;
            }
            else
            {
                outputs[0]->attr.dim_num = valid_dim_num;
                for (i = 0; i < inputs[0]->attr.dim_num; i++)
                {
                    if (dim_map[i] == 0)
                    {
                        outputs[0]->attr.size[index] = inputs[0]->attr.size[i];
                        index++;
                    }
                }
                if (1 == outputs[0]->attr.dim_num)
                {
                    outputs[0]->attr.dim_num = 2;
                    outputs[0]->attr.size[1] = 1;
                }
            }
        }
    }

    if (FALSE == op_set_reduce_axis(self, inputs, shape, &new_rank))
    {
        VSILOGE("op_set_reduce_axis error");
        return FALSE;
    }
    reshape_in_t[0] = vsi_nn_reshape_tensor( self->graph,
            inputs[0], (uint32_t*)shape, new_rank );

    self->nn_param.reduce.local2->reshaped_input1 = reshape_in_t[0];
    for (j = 0; j < self->nn_param.reduce.local2->axes_num; j++)
    {
        shape[self->nn_param.reduce.local2->axes[j]] = 1;
    }

    reshape_out_t[0] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], (uint32_t*)shape, new_rank );
    self->nn_param.reduce.local2->reshaped_output1 = reshape_out_t[0];
    if (self->nn_param.reduce.type == VSI_NN_REDUCE_SUM)
    {
        ret = op_set_reduce_internal(self, reshape_in_t, reshape_out_t, VSI_NN_OP_REDUCESUM_INTERNAL);
    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_MAX)
    {
        ret = op_set_reduce_internal(self, reshape_in_t, reshape_out_t, VSI_NN_OP_REDUCEMAX_INTERNAL);
    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_MIN)
    {
        ret = op_set_reduce_internal(self, reshape_in_t, reshape_out_t, VSI_NN_OP_REDUCEMIN_INTERNAL);
    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_PROD)
    {
        ret = op_set_reduce_internal(self, reshape_in_t, reshape_out_t, VSI_NN_OP_REDUCEPROD_INTERNAL);
    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_ALL)
    {
        ret = op_set_reduce_internal(self, reshape_in_t, reshape_out_t, VSI_NN_OP_REDUCEALL_INTERNAL);
    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_ANY)
    {
        ret = op_set_reduce_internal(self, reshape_in_t, reshape_out_t, VSI_NN_OP_REDUCEANY_INTERNAL);
    }


    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.reduce.local.axis_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local.axis_tensor));
    }

    if (self->nn_param.reduce.local2 != NULL)
    {
        if (self->nn_param.reduce.local2->axis_tensor2 != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local2->axis_tensor2));
        }
        if (self->nn_param.reduce.local2->reshaped_tmp != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local2->reshaped_tmp));
        }
        if (self->nn_param.reduce.local2->reshaped_output != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local2->reshaped_output));
        }
        if (self->nn_param.reduce.local2->reshaped_input != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local2->reshaped_input));
        }
        if (self->nn_param.reduce.local2->reshaped_output1 != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local2->reshaped_output1));
        }
        if (self->nn_param.reduce.local2->reshaped_input1 != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local2->reshaped_input1));
        }
        free(self->nn_param.reduce.local2);
        self->nn_param.reduce.local2 = NULL;
    }

    if (self->nn_param.reduce.type == VSI_NN_REDUCE_SUM ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_MAX ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_MIN ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_ALL ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_ANY ||
        self->nn_param.reduce.type == VSI_NN_REDUCE_PROD)
    {
        vsi_nn_internal_deinit_node_wksp(self);
    }
    else
    {
        vsi_nn_op_common_deinit(self);
    }

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    self->nn_param.reduce.local2   =
    (vsi_nn_reduce_lcl2_data_t *)malloc(sizeof(vsi_nn_reduce_lcl2_data_t));
    if (NULL == self->nn_param.reduce.local2)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(self->nn_param.reduce.local2, 0, sizeof(vsi_nn_reduce_lcl2_data_t));
    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REDUCE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

