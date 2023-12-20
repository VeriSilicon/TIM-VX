/****************************************************************************
*
*    Copyright 2017 - 2020 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef __VX_KHR_COMPATIBLE_H__
#define __VX_KHR_COMPATIBLE_H__
/*
 VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS is used to distingush deconvolution weight layout
 [value]
 0: weight_layout is whnc
 1: weight_layout is whcn
*/
#ifndef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
#define VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS 1
#endif
/*
 VX_CONVERT_POLICY_WRAP_ENABLE is used to differentiate two overflow_policys(VX_CONVERT_POLICY_WRAP and VX_CONVERT_POLICY_SAT)
 [value]
 0: both overflow_policys considered as VX_CONVERT_POLICY_SAT
 1: overflow_policy is determined by arguments.
*/
#ifndef VX_CONVERT_POLICY_WRAP_ENABLE
#define VX_CONVERT_POLICY_WRAP_ENABLE 1
#endif

#ifndef VX_13_NN_COMPATIBLITY
#define VX_13_NN_COMPATIBLITY 1
#endif
/*
 VX_L2NORM_AXIS_PARAMETER_SUPPORT is used to declare that L2NORMALIZE can support axis parameter
 [value]
 0: not support
 1: support
*/
#ifndef VX_L2NORM_AXIS_PARAMETER_SUPPORT
#define VX_L2NORM_AXIS_PARAMETER_SUPPORT 1
#endif
/*
 VX_SOFTMAX_AXIS_PARAMETER_SUPPORT is used to declare that SOFTAMX can support axis parameter
 [value]
 0: not support
 1: support
*/
#ifndef VX_SOFTMAX_AXIS_PARAMETER_SUPPORT
#define VX_SOFTMAX_AXIS_PARAMETER_SUPPORT 1
#endif
/*
 VX_NORMALIZATION_AXIS_PARAMETER_SUPPORT is used to declare that NORMALIZATION can support axis parameter
 [value]
 0: not support
 1: support
*/
#ifndef VX_NORMALIZATION_AXIS_PARAMETER_SUPPORT
#define VX_NORMALIZATION_AXIS_PARAMETER_SUPPORT 1
#endif
/*
 VX_ACTIVATION_EXT_SUPPORT is used to declare that ACTIVATION can support swish and hswish
 [value]
 0: not support
 1: support
*/
#ifndef VX_ACTIVATION_EXT_SUPPORT
#define VX_ACTIVATION_EXT_SUPPORT 1
#endif

/*
 VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT is used to query more hardware parameter such as shader sub-group size.
 [value]
 0: not support
 1: support
*/
#ifndef VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT
#define VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT 1
#endif

/*
 VX_VA40_EXT_SUPPORT is used to declare that openvx can support VA40.
 [value]
 0: not support
 1: support
*/
#ifndef VX_VA40_EXT_SUPPORT
#define VX_VA40_EXT_SUPPORT 0
#endif

/*
 VX_USER_LOOKUP_TABLE_SUPPORT is used to declare that openvx can support user lookuptable.
 [value]
 0: not support
 1: support
*/
#ifndef VX_USER_LOOKUP_TABLE_SUPPORT
#define VX_USER_LOOKUP_TABLE_SUPPORT 1
#endif

/*
VX_PRELOAD_CONST_TENSOR_SUPPORT is used to declare that openvx can support preload weight/bias and const tensor
 [value]
 0: not support
 1: support(NN conv and TP FC weightbias, and SH const tensor)
*/
#ifndef VX_PRELOAD_CONST_TENSOR_SUPPORT
#define VX_PRELOAD_CONST_TENSOR_SUPPORT 1
#endif

/*
VX_CREATE_TENSOR_SUPPORT_PHYSICAL is used to declare that openvx can support physical address for vxCreateTensorFromHandle
 [value]
 0: not support
 1: support
*/
#ifndef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
#define VX_CREATE_TENSOR_SUPPORT_PHYSICAL 1
#endif

/*
 VX_GRAPH_PREEMPTION_SUPPORT is used to declare that openvx can support different graph preemption function.
 [value]
 0: not support
 1: support
*/
#ifndef VX_GRAPH_PREEMPTION_SUPPORT
#define VX_GRAPH_PREEMPTION_SUPPORT 1
#endif

/*
VX_BATCH_GEMM_API_SUPPORT is used to declare that vsi openvx driver can support vxBatchGemmNode API to transform gemm to convolution
 [value]
 0: not support
 1: support
*/
#ifndef VX_BATCH_GEMM_API_SUPPORT
#define VX_BATCH_GEMM_API_SUPPORT 1
#endif

/*
VX_CONV_3D_API_SUPPORT is used to declare that vsi openvx driver can support conv3d by vxConv3dLayer API.
 [value]
 0: not support
 1: support
*/
#ifndef VX_CONV_3D_API_SUPPORT
#define VX_CONV_3D_API_SUPPORT 1
#endif

/*
VX_DECONV_3D_API_SUPPORT is used to declare that vsi openvx driver can support deconv3d by vxDeconv3dLayer API.
 [value]
 0: not support
 1: support
*/
#ifndef VX_DECONV_3D_API_SUPPORT
#define VX_DECONV_3D_API_SUPPORT 1
#endif

/*
 VX_PAD_CONST_SUPPORT is used to declare that openvx can support pad_const for tensorpad and convolution.
 [value]
 0: not support
 1: support
*/
#ifndef VX_PAD_CONST_SUPPORT
#define VX_PAD_CONST_SUPPORT 1
#endif

/*
 VX_TENSOR_STRIDE_X_BITS_SUPPORT is used to declare that openvx can support tensor which bits of stride in x dimension is not an integer number of bytes.
 [value]
 0: not support
 1: support
*/
#ifndef VX_TENSOR_STRIDE_X_BITS_SUPPORT
#define VX_TENSOR_STRIDE_X_BITS_SUPPORT 1
#endif

/*
VX_REMOVE_RESHAPE_SUPPORT is used to declare if graph opt support to remove reshape op, if support, it's not need to remove reshape in ovxlib.
 0: not support
 1: support
*/
/*
#ifndef VX_REMOVE_RESHAPE_SUPPORT
#define VX_REMOVE_RESHAPE_SUPPORT 0
#endif
*/

/*
VX_STREAM_PROCESSOR_SUPPORT is used to declare that vsi openvx driver can support vxStreamProcessorNode API
 [value]
 0: not support
 1: support
*/
#ifndef VX_STREAM_PROCESSOR_SUPPORT
#define VX_STREAM_PROCESSOR_SUPPORT 1
#endif

/*
 VX_TENSOR_MEMORY_CONNECT_DMA_CHANNEL is used to declare that this tensor connect to fixed DMA channel.
 [value]
 0: not support
 1: support
*/
#ifndef VX_TENSOR_MEMORY_CONNECT_DMA_CHANNEL
#define VX_TENSOR_MEMORY_CONNECT_DMA_CHANNEL 1
#endif

/*
 VX_SCALE_EXTRA_PARAMETER_SUPPORT is used to declare that RESIZE can support align_cornor and half_pixel_center parameter
 [value]
 0: not support
 1: support
*/
#ifndef VX_SCALE_EXTRA_PARAMETER_SUPPORT
#define VX_SCALE_EXTRA_PARAMETER_SUPPORT 1
#endif

/*
 VX_INVALIDATE_HANDLE_SUPPORT is used to declare that we refined vxSwapTensorHandle API to follow KHR OpenVX 1.3 spec: tensor don't maintain handle internally if new_ptr is NULL.
 [value]
 0: not support
 1: support
*/
#ifndef VX_INVALIDATE_HANDLE_SUPPORT
#define VX_INVALIDATE_HANDLE_SUPPORT 1
#endif

/*
 VX_ACTIVATION_EXT2_SUPPORT is used to declare that ACTIVATION can support sign, hard_sigmoid, neg, clip, exp, sin, cos,
 log, mish, gelu, hgelu, elu, selu, celu, rcp, softsign, atan, atanh, acosh, inverse sigmoid, round and erf.
 [value]
 0: not support
 1: support
*/
#ifndef VX_ACTIVATION_EXT2_SUPPORT
#define VX_ACTIVATION_EXT2_SUPPORT 1
#endif

/*
 VX_TENSORVIEW_ON_ANY_DIM is used to declare that ovxlib can do optimization for all concat node(all dimision) to tensor view if possiable, not only channel.
 [value]
 0: disable
 1: enable
*/
#ifndef VX_TENSORVIEW_ON_ANY_DIM
#define VX_TENSORVIEW_ON_ANY_DIM 0
#endif

/*
VX_DEPTH2SPACE_CRD_MODE_SUPPORT is used to declare that SPACE2DEPTH can support CRD mode
 [value]
 0: not support
 1: support
*/
#ifndef VX_DEPTH2SPACE_CRD_MODE_SUPPORT
#define VX_DEPTH2SPACE_CRD_MODE_SUPPORT 1
#endif

/*
 VX_LAYER_NORMALIZATION_VX_SUPPORT is used to declare driver support layer normalization layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_LAYER_NORMALIZATION_VX_SUPPORT
#define VX_LAYER_NORMALIZATION_VX_SUPPORT 1
#endif

/*
 VX_LAYER_NORMALIZATION_VX_SUPPORT is used to declare driver support layer normalization layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_INSTANCE_NORMALIZATION_VX_SUPPORT
#define VX_INSTANCE_NORMALIZATION_VX_SUPPORT 1
#endif

/*
 VX_GROUP_NORMALIZATION_VX_SUPPORT is used to declare driver support layer normalization layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_GROUP_NORMALIZATION_VX_SUPPORT
#define VX_GROUP_NORMALIZATION_VX_SUPPORT 1
#endif

/*
 VX_LOGICAL_VX_SUPPORT is used to declare driver support layer logical related layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_LOGICAL_VX_SUPPORT
#define VX_LOGICAL_VX_SUPPORT 1
#endif

/*
 VX_RELATIONAL_OPS_VX_SUPPORT is used to declare driver support layer relational related layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_RELATIONAL_OPS_VX_SUPPORT
#define VX_RELATIONAL_OPS_VX_SUPPORT 1
#endif

/*
 VX_REDUCE_MAX_VX_SUPPORT is used to declare driver support layer reduce max layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_REDUCE_MAX_VX_SUPPORT
#define VX_REDUCE_MAX_VX_SUPPORT 1
#endif

/*
 VX_REDUCE_MEAN_VX_SUPPORT is used to declare driver support layer reduce mean layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_REDUCE_MEAN_VX_SUPPORT
#define VX_REDUCE_MEAN_VX_SUPPORT 1
#endif

/*
 VX_REDUCE_SUM_VX_SUPPORT is used to declare driver support layer reduce sum layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_REDUCE_SUM_VX_SUPPORT
#define VX_REDUCE_SUM_VX_SUPPORT 1
#endif

/*
 VX_MAX_MIN_IMUM_VX_SUPPORT is used to declare driver support maximum and minimum layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_MAX_MIN_IMUM_VX_SUPPORT
#define VX_MAX_MIN_IMUM_VX_SUPPORT 1
#endif

/*
 VX_TENSOR_SELECR_VX_SUPPORT is used to declare driver support tensor select layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_TENSOR_SELECT_VX_SUPPORT
#define VX_TENSOR_SELECT_VX_SUPPORT 1
#endif

/*
 VX_GRU_CELL_VX_SUPPORT is used to declare driver support gru cell layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_GRU_CELL_VX_SUPPORT
#define VX_GRU_CELL_VX_SUPPORT 1
#endif

/*
 VX_LSTM_ACTIVATION_SUPPORT is used to declare driver support gru cell layer.
 [value]
 0: not support
 1: support
*/
#ifndef VX_LSTM_ACTIVATION_SUPPORT
#define VX_LSTM_ACTIVATION_SUPPORT 1
#endif

#endif /* __VX_KHR_COMPATIBLE_H__ */
