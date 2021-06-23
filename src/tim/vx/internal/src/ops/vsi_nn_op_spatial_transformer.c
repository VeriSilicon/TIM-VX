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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_dtype_util.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)
#define _VSI_PARAM          (vsi_nn_spatial_transformer_param)

extern vx_kernel_description_t * vx_kernel_SPATIAL_TRANSFORMER_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for ( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for ( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */


static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_context ctx;
    vsi_nn_spatial_transformer_param * p;
    int32_t  flag = 0;
    vsi_nn_tensor_t * thre_tensor;
    vsi_nn_tensor_attr_t attr;

    uint16_t value_buf[6] = {0};

    if ( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = (vsi_nn_spatial_transformer_param *)node->nn_param.client_param;
    ctx = vxGetContext( (vx_reference)node->graph->g );

    flag = ((p->has_theta_1_1 == 1)
            | ((p->has_theta_1_2 == 1) << 1)
            | ((p->has_theta_1_3 == 1) << 2)
            | ((p->has_theta_2_1 == 1) << 3)
            | ((p->has_theta_2_2 == 1) << 4)
            | ((p->has_theta_2_3 == 1) << 5));

    params[0] = (vx_reference)vxCreateScalar( ctx, VSI_NN_TYPE_INT32, &flag );

    memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    attr.size[0] = 6;
    attr.size[1] = 1;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 4;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;

    vsi_nn_Float32ToDtype(p->theta_1_1, (uint8_t*)(&value_buf[0]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_1_2, (uint8_t*)(&value_buf[1]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_1_3, (uint8_t*)(&value_buf[2]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_1, (uint8_t*)(&value_buf[3]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_2, (uint8_t*)(&value_buf[4]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_3, (uint8_t*)(&value_buf[5]), &attr.dtype);

    thre_tensor = vsi_nn_CreateTensorFromData( node->graph,(uint8_t *)&value_buf, &attr );

    params[1] = (vx_reference)thre_tensor->t;
    p->lcl.local_tensor = thre_tensor;
    p->lcl.scl = (vx_scalar)params[0];
#if 0
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VSI_NN_TYPE_FLOAT32, has_theta_1_3 );
    _SET_PARAM( 1, VSI_NN_TYPE_FLOAT32, has_theta_2_1 );
    _SET_PARAM( 2, VSI_NN_TYPE_FLOAT32, has_theta_1_2 );
    _SET_PARAM( 3, VSI_NN_TYPE_FLOAT32, theta_2_1 );
    _SET_PARAM( 4, VSI_NN_TYPE_FLOAT32, has_output_W );
    _SET_PARAM( 5, VSI_NN_TYPE_INT32, output_W );
    _SET_PARAM( 6, VSI_NN_TYPE_FLOAT32, theta_1_3 );
    _SET_PARAM( 7, VSI_NN_TYPE_FLOAT32, theta_2_2 );
    _SET_PARAM( 8, VSI_NN_TYPE_FLOAT32, theta_1_2 );
    _SET_PARAM( 9, VSI_NN_TYPE_INT32, output_H );
    _SET_PARAM( 10, VSI_NN_TYPE_FLOAT32, has_theta_2_3 );
    _SET_PARAM( 11, VSI_NN_TYPE_FLOAT32, theta_2_3 );
    _SET_PARAM( 12, VSI_NN_TYPE_FLOAT32, has_theta_2_2 );
    _SET_PARAM( 13, VSI_NN_TYPE_FLOAT32, has_output_H );
    _SET_PARAM( 14, VSI_NN_TYPE_FLOAT32, has_theta_1_1 );
    _SET_PARAM( 15, VSI_NN_TYPE_FLOAT32, theta_1_1 );
    #undef _SET_PARAM
#endif
//set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_nn_spatial_transformer_param * p = NULL;

    p = (vsi_nn_spatial_transformer_param *)node->nn_param.client_param;

    if (p->lcl.local_tensor) vsi_nn_ReleaseTensor(&p->lcl.local_tensor);
    if (p->lcl.scl) vxReleaseScalar(&p->lcl.scl);
} /* _release_params() */

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( self, args, _ARG_NUM );

    return status;
}

vsi_status setUPGridData(uint32_t output_W_, uint32_t output_H_, float scale, int32_t zeropoint,
         vsi_nn_dtype_t data_type, vsi_nn_qnt_type_e qnt_type, uint8_t fp, int16_t *tensorData)
{
    vsi_status   status = VSI_SUCCESS;
    uint32_t x               = 0;
    uint32_t y               = 0;
    uint32_t idx             = 0;
    float *tmp_buf = NULL;
    uint32_t i = 0;
    vsi_nn_dtype_t dtype;

    dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    dtype.fl = 0;
    dtype.scale = 1;
    dtype.zero_point = 0;

    tmp_buf = (float*) malloc(output_W_ * output_H_ * 3 * sizeof(float));
    if ( tmp_buf == NULL )
    {
         return VX_FAILURE;
    }
    for (y = 0; y < output_H_; y++ )
    {
        for (x = 0; x < output_W_; x++)
        {
            float data0 = y * (float)1.0 / (float)output_H_ * 2 - 1;
            float data1 = x * (float)1.0 / (float)output_W_ * 2 - 1;
            float data2 = 1;

            tmp_buf[idx++] = data0;
            tmp_buf[idx++] = data1;
            tmp_buf[idx++] = data2;
        }
    }

    for (i = 0; i < output_H_ * output_W_ * 3; i++)
    {
        vsi_nn_Float32ToDtype( tmp_buf[i], (uint8_t*)&tensorData[i], &dtype );
    }

    if (tmp_buf)
    {
        free(tmp_buf);
        tmp_buf = NULL;
    }
    return status;
}

static vsi_status vx_op_compute_setupThre
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[4] = {NULL};
    //vx_reference * args;
    vsi_nn_spatial_transformer_param * p = NULL;
    int flag = 0;
    vsi_nn_tensor_t * thre_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    vx_context ctx = NULL;
    vx_scalar flag_s = NULL;
    vx_tensor tmp_t = NULL, tmp_t1 = NULL;

    //float flag_buf[6];
    vx_uint16 value_buf[6];

    memset( params, 0, sizeof( vx_reference * ) * 4 );
    p = (vsi_nn_spatial_transformer_param *)self->nn_param.client_param;
    ctx = vxGetContext( (vx_reference)self->graph->g );

    flag = ((p->has_theta_1_1 == 1)
            | ((p->has_theta_1_2 == 1) << 1)
            | ((p->has_theta_1_3 == 1) << 2)
            | ((p->has_theta_2_1 == 1) << 3)
            | ((p->has_theta_2_2 == 1) << 4)
            | ((p->has_theta_2_3 == 1) << 5));

    memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    attr.size[0] = 6;
    attr.size[1] = 1;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 4;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.fl = 0;
    attr.dtype.scale = 1;
    attr.dtype.zero_point = 0;
    attr.vtl = FALSE;
    vsi_nn_Float32ToDtype( p->theta_1_1, (uint8_t*)(&value_buf[0]), &attr.dtype );
    vsi_nn_Float32ToDtype( p->theta_1_2, (uint8_t*)(&value_buf[1]), &attr.dtype );
    vsi_nn_Float32ToDtype( p->theta_1_3, (uint8_t*)(&value_buf[2]), &attr.dtype );
    vsi_nn_Float32ToDtype( p->theta_2_1, (uint8_t*)(&value_buf[3]), &attr.dtype );
    vsi_nn_Float32ToDtype( p->theta_2_2, (uint8_t*)(&value_buf[4]), &attr.dtype );
    vsi_nn_Float32ToDtype( p->theta_2_3, (uint8_t*)(&value_buf[5]), &attr.dtype );

    thre_tensor = vsi_nn_CreateTensorFromData( self->graph,(uint8_t *)&value_buf, &attr );

    if ( NULL == self->n )
    {
        status = VSI_FAILURE;
        if (thre_tensor)
        {
            vsi_nn_ReleaseTensor( &thre_tensor);
            thre_tensor = NULL;
        }
        return status;
    }

    flag_s = vxCreateScalar( ctx, VSI_NN_TYPE_INT32, &flag );

    params[0] = (vx_reference)thre_tensor->t;

    attr.size[0] = inputs[0]->attr.size[0] * inputs[0]->attr.size[1];
    attr.size[1] = 1;
    attr.size[2] = inputs[0]->attr.size[2];
    attr.size[3] = inputs[0]->attr.size[3];
    attr.dim_num = inputs[0]->attr.dim_num;

    tmp_t = vxReshapeTensor( inputs[0]->t, (vx_int32*)attr.size, attr.dim_num );

    params[1] = (vx_reference)tmp_t;
    params[2] = (vx_reference)flag_s;

    attr.size[0] = outputs[0]->attr.size[0] * outputs[0]->attr.size[1];
    attr.size[1] = 1;
    attr.size[2] = outputs[0]->attr.size[2];
    attr.size[3] = outputs[0]->attr.size[3];
    attr.dim_num = outputs[0]->attr.dim_num;

    tmp_t1 = vxReshapeTensor( outputs[0]->t, (vx_int32*)attr.size, attr.dim_num );

    params[3] = (vx_reference)tmp_t1;

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 4 );

    //_release_params( args, 4 );
    if (thre_tensor)
    {
        vsi_nn_ReleaseTensor( &thre_tensor);
        thre_tensor = NULL;
    }
    if (tmp_t)
    {
        vxReleaseTensor( &tmp_t );
        tmp_t = NULL;
    }
    if (tmp_t1)
    {
        vxReleaseTensor( &tmp_t1 );
        tmp_t1 = NULL;
    }
    if (flag_s)
    {
        vxReleaseScalar( &flag_s );
        flag_s = NULL;
    }

    return status;
}

static vsi_status vx_op_compute_gemm
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status   status = VSI_SUCCESS;
    vx_reference params[3] = {NULL};
    vx_tensor paraTensor0 = NULL, paraTensor1 = NULL, paraTensor2 = NULL;
    int32_t     size[4]    = {1};
    vsi_nn_tensor_attr_t out_attr;
    int16_t *out_buffer = NULL;
    uint32_t output_H = 0, output_W = 0;
    float   *buf = NULL;

    memcpy( &out_attr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
    output_W = out_attr.size[0];
    output_H = out_attr.size[1];
    out_buffer = (int16_t*)malloc( output_W * output_H * 3 * sizeof(int16_t) );
    status = setUPGridData( output_W, output_H, out_attr.dtype.scale, out_attr.dtype.zero_point,
        out_attr.dtype, out_attr.dtype.qnt_type, out_attr.dtype.fl, out_buffer );
    if (status == VSI_FAILURE)
    {
        goto OnError;
    }
    status = vsi_nn_copy_tensor_patch( inputs[1]->t, &inputs[1]->attr, out_buffer, VX_WRITE_ONLY );
    if (status == VSI_FAILURE)
    {
        goto OnError;
    }
    /* Copy tensor to buffer, and convert bufer to float32 format */
    buf = vsi_nn_ConvertTensorToFloat32Data(self->graph, inputs[1]);
    if (buf == NULL)
    {
        goto OnError;
    }
    memset( params, 0, sizeof( vx_reference * ) * 3 );

    size[0] = inputs[0]->attr.size[0] * inputs[0]->attr.size[1];
    size[1] = 1;
    paraTensor0 = vxReshapeTensor( inputs[0]->t, size, 2 );

    size[0] = inputs[1]->attr.size[0] * output_W;
    size[1] = output_H;
    paraTensor1 = vxReshapeTensor( inputs[1]->t, size, 2 );

    size[0] = inputs[0]->attr.size[1] * output_W;
    size[1] = output_H;
    paraTensor2 = vxReshapeTensor( inputs[2]->t, size, 2 );

    if ( NULL == self->n )
    {
        status = VSI_FAILURE;
        goto OnError;
    }

    params[0] = (vx_reference)paraTensor0;
    params[1] = (vx_reference)paraTensor1;
    params[2] = (vx_reference)paraTensor2;
    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 3 );

OnError:
    if (paraTensor0)
    {
        vxReleaseTensor( &paraTensor0 );
        paraTensor0 = NULL;
    }
    if (paraTensor1)
    {
        vxReleaseTensor( &paraTensor1 );
        paraTensor1 = NULL;
    }
    if (paraTensor2)
    {
        vxReleaseTensor( &paraTensor2 );
        paraTensor2 = NULL;
    }
    if (out_buffer)
    {
        free(out_buffer);
        out_buffer = NULL;
    }
    return status;
}


static vsi_status vx_op_compute_interp
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[3];
    vx_border_t border;

    memset( params, 0, sizeof( vx_reference * ) * 3 );

    params[0] = (vx_reference)inputs[3]->t;
    params[1] = (vx_reference)inputs[2]->t;
    params[2] =(vx_reference)outputs[0]->t;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 3 );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.S16 = 0;

    status |= vxSetNodeAttribute( self->n, VX_NODE_BORDER,
        &border, sizeof(border) );
   // _release_params( args, 3 );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute_setupThre,
    vx_op_compute_gemm,
    vx_op_compute_interp,
    NULL
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_kernel_info_t kernel_info;
    char *path = NULL;
    vsi_nn_tensor_attr_t attr,outattr;
    vsi_nn_tensor_t *tmp_output_tensor[5] = {0};
    vsi_nn_tensor_t *input_t,*fc_t,*output_t;
    vx_graph graph = self->graph->g;

    memset( &kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t) );
    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );

    memcpy( &attr, &(inputs[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.fl = 0;
    attr.dtype.scale = 1;
    attr.dtype.zero_point = 0;
    attr.vtl = FALSE;

    input_t = vsi_nn_CreateTensor( self->graph, &attr );

    memcpy( &attr, &(inputs[1]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.fl = 0;
    attr.dtype.scale = 1;
    attr.dtype.zero_point = 0;
    attr.vtl = FALSE;
    fc_t= vsi_nn_CreateTensor( self->graph, &attr );

    memcpy( &attr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.fl = 0;
    attr.dtype.scale = 1;
    attr.dtype.zero_point = 0;
    attr.vtl = FALSE;
    output_t= vsi_nn_CreateTensor( self->graph, &attr );

    vxTensorCopyNode( graph, inputs[0]->t, input_t->t );
    vxTensorCopyNode( graph, inputs[1]->t, fc_t->t );
    vxTensorCopyNode( graph, output_t->t, outputs[0]->t );

    memcpy( &outattr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t) );
     // Tensor for thre_output
    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.size[0] = 3;
    attr.size[1] = 2;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    tmp_output_tensor[0] = vsi_nn_CreateTensor( self->graph, &attr );

    // Tensor for grid
    attr.size[0] = 3;
    attr.size[1] = outattr.size[0] * outattr.size[1];//p->output_H * p->output_W;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    tmp_output_tensor[1] = vsi_nn_CreateTensor( self->graph, &attr );

    // Tensor for grid_out
    attr.size[0] = 2 * outattr.size[0];//2 * p->output_W;
    attr.size[1] = outattr.size[1];//p->output_H ;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    tmp_output_tensor[2] = vsi_nn_CreateTensor( self->graph, &attr );
    status = VSI_FAILURE;


    kernel_info.type = VX_KERNEL_TYPE_VX;
    kernel_info.kernel = vx_kernel_SPATIAL_TRANSFORMER_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc( kernel_info.resource_num * sizeof(char *) );
    kernel_info.resource_name[0] = "vsi_nn_kernel_transform_setupThres";

    path = getenv("USER_VX_SOURCE_PATH");
    if(path)
        vsi_nn_VxResourceSetPath( path );

    kernel_info.kernel_index = 1;
    kernel_info.init_index = 1;

    // add setupThre
    self->n = vsi_nn_RegisterClientKernelAndNewNode( self->graph, &kernel_info);

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index]( self, &fc_t, tmp_output_tensor );
    }

    if ( NULL == self->n )
    {
        status = VSI_FAILURE;
        goto final;
    }

    // add gemm
    kernel_info.kernel_index = 2;
    kernel_info.init_index = 2;
    kernel_info.resource_name[0] = "vsi_nn_kernel_transform_gemm";
    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, tmp_output_tensor, outputs);
    }

    // add interp
    if (input_t->attr.dim_num == 2 && input_t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
            && output_t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info.kernel_index = 3;
        kernel_info.init_index = 3;
    }
    else if (input_t->attr.dim_num == 4 && input_t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
            && output_t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
    {
         kernel_info.kernel_index = 4;
         kernel_info.init_index = 3;
    }
    kernel_info.resource_name[0] = "vsi_nn_kernel_transform_interp";
    self->n = vsi_nn_RegisterClientKernelAndNewNode( self->graph, &kernel_info);
    tmp_output_tensor[3] = input_t;

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index]( self, tmp_output_tensor, &output_t );
    }
    if (tmp_output_tensor[0])
    {
        vsi_nn_ReleaseTensor( &tmp_output_tensor[0] );
        tmp_output_tensor[0] = NULL;
    }
    if (tmp_output_tensor[1])
    {
        vsi_nn_ReleaseTensor( &tmp_output_tensor[1] );
        tmp_output_tensor[1] = NULL;
    }
    if (tmp_output_tensor[2])
    {
        vsi_nn_ReleaseTensor( &tmp_output_tensor[2] );
        tmp_output_tensor[2] = NULL;
    }
    if (input_t)
    {
        vsi_nn_ReleaseTensor( &input_t );
        input_t = NULL;
    }
    if (fc_t)
    {
         vsi_nn_ReleaseTensor( &fc_t );
         fc_t = NULL;
    }
    if (output_t)
    {
        vsi_nn_ReleaseTensor( &output_t );
        output_t = NULL;
    }

final:
    if (kernel_info.resource_name)
    {
        free(kernel_info.resource_name);
        kernel_info.resource_name = NULL;
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
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    //vsi_nn_spatial_transformer_param * p;
    //p = (vsi_nn_spatial_transformer_param *)&node->nn_param.client_param;

    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    outputs[0]->attr.size[0] = inputs[0]->attr.size[0];//p->output_W;  // W
    outputs[0]->attr.size[1] = inputs[0]->attr.size[1];//p->output_H;  // H
    outputs[0]->attr.size[2] = inputs[0]->attr.size[2]; // C
    outputs[0]->attr.size[3] = inputs[0]->attr.size[3]; // N
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SPATIAL_TRANSFORMER,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
