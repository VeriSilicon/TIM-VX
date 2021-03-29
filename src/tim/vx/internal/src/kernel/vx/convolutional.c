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

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_node.h"
#include "vsi_nn_feature.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_graph_optimization.h"

static vsi_bool _build_vx_conv2d_param
    (
    vx_nn_convolution_params_ext2_t * param,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h_front, int32_t pad_h_end,
    int32_t pad_w_front, int32_t pad_w_end,
    int32_t dilation_h, int32_t dilation_w,
    int32_t multiplier,
    vsi_enum overflow_policy, vsi_enum rounding_policy,
    vsi_enum down_scale_size_rounding
    )
{
    vx_nn_convolution_params_ext_t * p1 = NULL;
    memset( param, 0 ,sizeof(vx_nn_convolution_params_ext2_t) );

    VSI_ASSERT( stride_h > 0 );
    VSI_ASSERT( stride_w > 0 );
    VSI_ASSERT( pad_h_front >= 0 );
    VSI_ASSERT( pad_h_end >= 0 );
    VSI_ASSERT( pad_w_front >= 0 );
    VSI_ASSERT( pad_w_end >= 0 );
    VSI_ASSERT( dilation_h >= 0 );
    VSI_ASSERT( dilation_w >= 0 );
    VSI_ASSERT( multiplier >= 0 );

    p1 = &param->ext;
    p1->khr.padding_x = (uint32_t)pad_w_front;
    p1->khr.padding_y = (uint32_t)pad_h_front;
    if( dilation_h > 0 )
    {
        p1->khr.dilation_y = (uint32_t)(dilation_h - 1);
    }
    if( dilation_w > 0 )
    {
        p1->khr.dilation_x = (uint32_t)(dilation_w - 1);
    }
    //VSILOGD("pad %d %d %d %d", pad_h_front, pad_h_end, pad_w_front, pad_w_end);
    //VSILOGD("dilation %d %d ", p1->khr.dilation_y, p1->khr.dilation_x);
    //VSILOGD("mul %d ", multiplier);
    p1->khr.overflow_policy = (vx_enum)overflow_policy;
    p1->khr.rounding_policy = (vx_enum)rounding_policy;
    p1->khr.down_scale_size_rounding = (vx_enum)down_scale_size_rounding;
    p1->padding_x_right = (uint32_t)pad_w_end;
    p1->padding_y_bottom = (uint32_t)pad_h_end;
    param->depth_multiplier = multiplier;
    param->stride_x = (uint32_t)stride_w;
    param->stride_y = (uint32_t)stride_h;
    return TRUE;
} /* _build_vx_conv2d_param() */

static vsi_bool _build_vx_deconv2d_param
    (
    vx_nn_deconvolution_params_ext2_t * param,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h_front, int32_t pad_h_end,
    int32_t pad_w_front, int32_t pad_w_end,
    uint32_t group,
    vsi_enum overflow_policy, vsi_enum rounding_policy,
    vsi_enum down_scale_size_rounding
    )
{
    vx_nn_deconvolution_params_ext_t * p1 = NULL;
    memset( param, 0 ,sizeof(vx_nn_deconvolution_params_ext2_t) );

    VSI_ASSERT( stride_h > 0 );
    VSI_ASSERT( stride_w > 0 );
    VSI_ASSERT( pad_h_front >= 0 );
    VSI_ASSERT( pad_h_end >= 0 );
    VSI_ASSERT( pad_w_front >= 0 );
    VSI_ASSERT( pad_w_end >= 0 );
    VSI_ASSERT( group >= 1 );

    p1 = &param->ext;
    p1->khr.padding_x = (uint32_t)pad_w_front;
    p1->khr.padding_y = (uint32_t)pad_h_front;
    p1->khr.overflow_policy = (vx_enum)overflow_policy;
    p1->khr.rounding_policy = (vx_enum)rounding_policy;
    p1->padding_x_right = (uint32_t)pad_w_end;
    p1->padding_y_bottom = (uint32_t)pad_h_end;
    p1->channel_group = (uint32_t)group;
    param->stride_x = (uint32_t)stride_w;
    param->stride_y = (uint32_t)stride_h;
    param->down_scale_size_rounding = (vx_enum)down_scale_size_rounding;
    return TRUE;
} /* _build_vx_deconv2d_param() */

static vx_tensor _expand_tensor_dim
    ( vx_tensor tensor, int32_t * shape, size_t rank, int32_t expand_dim )
{
    int32_t new_shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t i, cnt;
    if( expand_dim < 0 )
    {
        expand_dim = (int32_t)rank + expand_dim;
    }
    if( expand_dim < 0 || (uint32_t)expand_dim > rank )
    {
        VSILOGE("Run dim to expand %d, rank is %lu", expand_dim, rank);
        return NULL;
    }
    for( i = 0, cnt = 0; i < rank; i ++ )
    {
        if( i == (uint32_t)expand_dim )
        {
            new_shape[cnt] = 1;
            cnt ++;
        }
        new_shape[cnt] = shape[i];
        cnt ++;
    }
    if( (uint32_t)expand_dim == rank )
    {
        new_shape[cnt] = 1;
    }
    return vxReshapeTensor( tensor, new_shape, (uint32_t)rank + 1 );
} /* _expand_tensor_dim() */


#define REGISTER_CONV_OPENVX_KERNEL( kernel_name )   \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ); \
    REGISTER_BACKEND_OPENVX( kernel_name, _##kernel_name##setup ) \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        )


REGISTER_CONV_OPENVX_KERNEL( conv1d )
{
    vx_node node = NULL;
    vx_nn_convolution_params_ext2_t vxparam;
    vx_tensor temp_tensors[3] = { NULL };
    int i;

    _build_vx_conv2d_param(
            &vxparam,
            vsi_nn_kernel_param_get_int32(params, "stride"), 1,
            vsi_nn_kernel_param_get_int32(params, "pad_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_end"),
            0,0,
            vsi_nn_kernel_param_get_int32(params, "dilation"), 1,
            0,
            vsi_nn_kernel_param_get_int32(params, "overflow_policy"),
            vsi_nn_kernel_param_get_int32(params, "rounding_policy"),
            vsi_nn_kernel_param_get_int32(params, "down_scale_size_rounding")
            );

    temp_tensors[0] = _expand_tensor_dim( inputs[0]->t,
            (int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num, 0 );
    CHECK_PTR_FAIL_GOTO( temp_tensors[0], "Expand input dim fail.", final );
    if (inputs[1]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC)
    {
        temp_tensors[1] = _expand_tensor_dim( inputs[1]->t,
                (int32_t*)inputs[1]->attr.size, inputs[1]->attr.dim_num, 0 );
        CHECK_PTR_FAIL_GOTO( temp_tensors[1], "Expand kernel dim fail.", final );
    }
    else
    {
        uint8_t    * data = NULL;
        vsi_nn_tensor_attr_t attr;
        uint32_t i;

        data = vsi_nn_ConvertTensorToData( graph, inputs[1] );
        CHECK_PTR_FAIL_GOTO( data, "Convert data fail.", final );

        memcpy(&attr, &inputs[1]->attr, sizeof(vsi_nn_tensor_attr_t));

        attr.size[0] = 1;
        for (i = 1; i <= inputs[1]->attr.dim_num; i++)
        {
            attr.size[i] = inputs[1]->attr.size[i - 1];
        }
        attr.dim_num = inputs[1]->attr.dim_num + 1;
        attr.dtype.channel_dim = inputs[1]->attr.dtype.channel_dim + 1;

        temp_tensors[1] = vsi_nn_CreateRawTensorFromData(graph, data, &attr);
        vsi_nn_safe_free( data );
    }

    temp_tensors[2] = _expand_tensor_dim( outputs[0]->t,
            (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num, 0 );
    CHECK_PTR_FAIL_GOTO( temp_tensors[2], "Expand output dim fail.", final );

    node = vxConvolutionLayer( graph->g,
        temp_tensors[0], temp_tensors[1], inputs[2] ? inputs[2]->t : NULL,
        (vx_nn_convolution_params_t *)&vxparam,
        sizeof( vx_nn_convolution_params_ext2_t ),
        temp_tensors[2]
        );

final:
    for( i = 0; i < 3; i ++ )
    {
        if( temp_tensors[i] )
        {
            vxReleaseTensor( &temp_tensors[i] );
        }
    }
    return (vsi_nn_kernel_node_t)node;
} /* conv1d*/

REGISTER_CONV_OPENVX_KERNEL( depthwise_conv1d )
{
    vx_node node = NULL;
    vx_nn_convolution_params_ext2_t vxparam;
    vx_tensor temp_tensors[3] = { NULL };
    int32_t i;
    vsi_bool need_explicit_padding = FALSE;

    _build_vx_conv2d_param(
            &vxparam,
            vsi_nn_kernel_param_get_int32(params, "stride"), 1,
            vsi_nn_kernel_param_get_int32(params, "pad_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_end"),
            0,0,
            vsi_nn_kernel_param_get_int32(params, "dilation"), 1,
            vsi_nn_kernel_param_get_int32(params, "multiplier"),
            vsi_nn_kernel_param_get_int32(params, "overflow_policy"),
            vsi_nn_kernel_param_get_int32(params, "rounding_policy"),
            vsi_nn_kernel_param_get_int32(params, "down_scale_size_rounding")
            );

    temp_tensors[0] = _expand_tensor_dim( inputs[0]->t,
            (int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num, 0 );
    CHECK_PTR_FAIL_GOTO( temp_tensors[0], "Expand input dim fail.", final );

    if (inputs[1]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC)
    {
        int32_t new_w_shape[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint32_t new_w_rank = 4;
        new_w_shape[0] = 1;
        new_w_shape[1] = inputs[1]->attr.size[0];
        new_w_shape[2] = 1;
        for (i = 1; i < (int32_t)(inputs[1]->attr.dim_num); i++)
        {
            new_w_shape[2] *= inputs[1]->attr.size[i];
        }
        new_w_shape[3] = 1;
        temp_tensors[1] = vxReshapeTensor( inputs[1]->t, new_w_shape, new_w_rank );
        CHECK_PTR_FAIL_GOTO( temp_tensors[1], "Expand kernel dim fail.", final );
    }
    else
    {
        uint8_t    * data = NULL;
        vsi_nn_tensor_attr_t attr;
        uint32_t i;

        data = vsi_nn_ConvertTensorToData( graph, inputs[1] );
        CHECK_PTR_FAIL_GOTO( data, "Convert data fail.", final );

        memcpy(&attr, &inputs[1]->attr, sizeof(vsi_nn_tensor_attr_t));

        attr.size[0] = 1;
        attr.size[1] = inputs[1]->attr.size[0];
        attr.size[2] = 1;
        for (i = 1; i < inputs[1]->attr.dim_num; i++)
        {
            attr.size[2] *= inputs[1]->attr.size[i];
        }
        attr.size[3] = 1;
        attr.dim_num = 4;
        attr.dtype.channel_dim = 2;

        temp_tensors[1] = vsi_nn_CreateRawTensorFromData(graph, data, &attr);

        vsi_nn_safe_free( data );
    }

    temp_tensors[2] = _expand_tensor_dim( outputs[0]->t,
            (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num, 0 );
    CHECK_PTR_FAIL_GOTO( temp_tensors[2], "Expand output dim fail.", final );

    if( need_explicit_padding )
    {
        int32_t pad_front[4] = { 0 };
        int32_t pad_end[4] = { 0 };
        vx_tensor pad_tensor = NULL;
        pad_front[0] = (int32_t)vxparam.ext.khr.padding_x;
        pad_front[1] = (int32_t)vxparam.ext.khr.padding_y;
        pad_end[0] = (int32_t)vxparam.ext.padding_x_right;
        pad_end[1] = (int32_t)vxparam.ext.padding_y_bottom;

        pad_tensor = (vx_tensor)kernel_pad_node(
                graph, (vsi_nn_kernel_tensor_t)temp_tensors[0],
                pad_front, pad_end, 4, VSI_NN_PAD_MODE_CONSTANT, 0, NULL);

        if( NULL == pad_tensor )
        {
            VSILOGW("Create pad node fail.");
            goto final;
        }
        else
        {
            vxReleaseTensor( &temp_tensors[0] );
            temp_tensors[0] = pad_tensor;
        }
        vxparam.ext.khr.padding_x = 0;
        vxparam.ext.khr.padding_y = 0;
        vxparam.ext.padding_x_right = 0;
        vxparam.ext.padding_y_bottom = 0;
    }

    node = vxConvolutionLayer( graph->g,
        temp_tensors[0], temp_tensors[1], inputs[2] ? inputs[2]->t : NULL,
        (vx_nn_convolution_params_t *)&vxparam,
        sizeof( vx_nn_convolution_params_ext2_t ),
        temp_tensors[2]
        );
final:
    for( i = 0; i < 3; i ++ )
    {
        if( temp_tensors[i] )
        {
            vxReleaseTensor( &temp_tensors[i] );
        }
    }
    return (vsi_nn_kernel_node_t)node;
} /* depthwise_conv1d*/

REGISTER_CONV_OPENVX_KERNEL( conv2d )
{
    vx_node node = NULL;
    vx_nn_convolution_params_ext2_t vxparam;

    _build_vx_conv2d_param(
            &vxparam,
            vsi_nn_kernel_param_get_int32(params, "stride_h"),
            vsi_nn_kernel_param_get_int32(params, "stride_w"),
            vsi_nn_kernel_param_get_int32(params, "pad_h_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_h_end"),
            vsi_nn_kernel_param_get_int32(params, "pad_w_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_w_end"),
            vsi_nn_kernel_param_get_int32(params, "dilation_h"),
            vsi_nn_kernel_param_get_int32(params, "dilation_w"),
            0,
            vsi_nn_kernel_param_get_int32(params, "overflow_policy"),
            vsi_nn_kernel_param_get_int32(params, "rounding_policy"),
            vsi_nn_kernel_param_get_int32(params, "down_scale_size_rounding")
            );

    node = vxConvolutionLayer( graph->g,
        inputs[0]->t, inputs[1]->t, inputs[2] ? inputs[2]->t : NULL,
        (vx_nn_convolution_params_t *)&vxparam,
        sizeof( vx_nn_convolution_params_ext2_t ),
        outputs[2]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* conv2d*/

REGISTER_CONV_OPENVX_KERNEL( depthwise_conv2d )
{
    vx_node node = NULL;
    vx_nn_convolution_params_ext2_t vxparam;

    _build_vx_conv2d_param(
            &vxparam,
            vsi_nn_kernel_param_get_int32(params, "stride_h"),
            vsi_nn_kernel_param_get_int32(params, "stride_w"),
            vsi_nn_kernel_param_get_int32(params, "pad_h_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_h_end"),
            vsi_nn_kernel_param_get_int32(params, "pad_w_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_w_end"),
            vsi_nn_kernel_param_get_int32(params, "dilation_h"),
            vsi_nn_kernel_param_get_int32(params, "dilation_w"),
            vsi_nn_kernel_param_get_int32(params, "multiplier"),
            vsi_nn_kernel_param_get_int32(params, "overflow_policy"),
            vsi_nn_kernel_param_get_int32(params, "rounding_policy"),
            vsi_nn_kernel_param_get_int32(params, "down_scale_size_rounding")
            );

    node = vxConvolutionLayer( graph->g,
        inputs[0]->t, inputs[1]->t, inputs[2] ? inputs[2]->t : NULL,
        (vx_nn_convolution_params_t *)&vxparam,
        sizeof( vx_nn_convolution_params_ext2_t ),
        outputs[2]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* depthwise_conv2d*/

REGISTER_CONV_OPENVX_KERNEL( deconvolution1d )
{
    vx_node node = NULL;
    vx_nn_deconvolution_params_ext2_t vxparam;
    vx_tensor temp_tensors[2] = { NULL };
    int i;

    _build_vx_deconv2d_param(
            &vxparam,
            1, vsi_nn_kernel_param_get_int32(params, "stride"),
            0,0,
            vsi_nn_kernel_param_get_int32(params, "pad_front"),
            vsi_nn_kernel_param_get_int32(params, "pad_end"),
            vsi_nn_kernel_param_get_int32(params, "group"),
            vsi_nn_kernel_param_get_int32(params, "overflow_policy"),
            vsi_nn_kernel_param_get_int32(params, "rounding_policy"),
            vsi_nn_kernel_param_get_int32(params, "down_scale_size_rounding")
            );

    temp_tensors[0] = _expand_tensor_dim( inputs[0]->t,
            (int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num, 1 );
    CHECK_PTR_FAIL_GOTO( temp_tensors[0], "Expand input dim fail.", final );

    temp_tensors[1] = _expand_tensor_dim( outputs[0]->t,
            (int32_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num, 1 );
    CHECK_PTR_FAIL_GOTO( temp_tensors[1], "Expand output dim fail.", final );

    node = vxDeconvolutionLayer( graph->g,
        temp_tensors[0], inputs[1]->t, inputs[2] ? inputs[2]->t : NULL,
        (vx_nn_deconvolution_params_t *)&vxparam,
        sizeof( vx_nn_deconvolution_params_ext2_t ),
        temp_tensors[1]
        );

final:
    for( i = 0; i < 2; i ++ )
    {
        if( temp_tensors[i] )
        {
            vxReleaseTensor( &temp_tensors[i] );
        }
    }
    return (vsi_nn_kernel_node_t)node;
} /* deconvolution1d*/

#undef REGISTER_CONV_OPENVX_KERNEL
