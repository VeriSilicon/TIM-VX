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
#define VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS 1
/*
 VX_CONVERT_POLICY_WRAP_ENABLE is used to differentiate two overflow_policys(VX_CONVERT_POLICY_WRAP and VX_CONVERT_POLICY_SAT)
 [value]
 0: both overflow_policys considered as VX_CONVERT_POLICY_SAT
 1: overflow_policy is determined by arguments.
*/
#define VX_CONVERT_POLICY_WRAP_ENABLE 1

#define VX_13_NN_COMPATIBLITY 1
/*
 VX_L2NORM_AXIS_PARAMETER_SUPPORT is used to declare that L2NORMALIZE can support axis parameter
 [value]
 0: not support
 1: support
*/
#define VX_L2NORM_AXIS_PARAMETER_SUPPORT 1
/*
 VX_SOFTMAX_AXIS_PARAMETER_SUPPORT is used to declare that SOFTAMX can support axis parameter
 [value]
 0: not support
 1: support
*/
#define VX_SOFTMAX_AXIS_PARAMETER_SUPPORT 1
/*
 VX_NORMALIZATION_AXIS_PARAMETER_SUPPORT is used to declare that NORMALIZATION can support axis parameter
 [value]
 0: not support
 1: support
*/
#define VX_NORMALIZATION_AXIS_PARAMETER_SUPPORT 1
/*
 VX_ACTIVATION_EXT_SUPPORT is used to declare that ACTIVATION can support swish and hswish
 [value]
 0: not support
 1: support
*/
#define VX_ACTIVATION_EXT_SUPPORT 1

/*
 VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT is used to query more hardware parameter such as shader sub-group size.
 [value]
 0: not support
 1: support
*/
#define VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT 1

/*
 VX_USER_LOOKUP_TABLE_SUPPORT is used to declare that openvx can support user lookuptable.
 [value]
 0: not support
 1: support
*/
#define VX_USER_LOOKUP_TABLE_SUPPORT 1

/*
VX_PRELOAD_CONST_TENSOR_SUPPORT is used to declare that openvx can support preload weight/bias and const tensor
 [value]
 0: not support
 1: support(NN conv and TP FC weightbias, and SH const tensor)
*/
#define VX_PRELOAD_CONST_TENSOR_SUPPORT 1

/*
VX_CREATE_TENSOR_SUPPORT_PHYSICAL is used to declare that openvx can support physical address for vxCreateTensorFromHandle
 [value]
 0: not support
 1: support
*/
#define VX_CREATE_TENSOR_SUPPORT_PHYSICAL 1

/*
 VX_GRAPH_PREEMPTION_SUPPORT is used to declare that openvx can support different graph preemption function.
 [value]
 0: not support
 1: support
*/
#define VX_GRAPH_PREEMPTION_SUPPORT 1

#endif /* __VX_KHR_COMPATIBLE_H__ */
