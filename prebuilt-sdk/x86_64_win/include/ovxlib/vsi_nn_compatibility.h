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
/** @file */
#ifndef _VSI_NN_COMPATIBILITY_H_
#define _VSI_NN_COMPATIBILITY_H_

#if defined(__cplusplus)
extern "C"{
#endif

/*
  keep the backward compatibility with spec 1.1 for standard nn kernels
*/
#define    VX_KERNEL_NN_SOFTMAX_LAYER                   VX_KERNEL_SOFTMAX_LAYER
#define    VX_KERNEL_NN_NORMALIZATION_LAYER             VX_KERNEL_NORMALIZATION_LAYER
#define    VX_KERNEL_NN_POOLING_LAYER                   VX_KERNEL_POOLING_LAYER
#define    VX_KERNEL_NN_FULLY_CONNECTED_LAYER           VX_KERNEL_FULLY_CONNECTED_LAYER
#define    VX_KERNEL_NN_ACTIVATION_LAYER                VX_KERNEL_ACTIVATION_LAYER
#define    VX_KERNEL_NN_ROIPOOL                         VX_KERNEL_ROI_POOLING_LAYER
#define    VX_KERNEL_NN_CONVOLUTION_LAYER               VX_KERNEL_CONVOLUTION_LAYER
#define    VX_KERNEL_NN_DECONVOLUTION_LAYER             VX_KERNEL_DECONVOLUTION_LAYER

/*
  keep the backward compatibility with spec 1.1 for vx_tensor_attribute_e
*/
#define    VX_TENSOR_NUM_OF_DIMS                        VX_TENSOR_NUMBER_OF_DIMS
#define    VX_TENSOR_FIXED_POINT_POS                    VX_TENSOR_FIXED_POINT_POSITION

/*
  keep the backward compatibility with spec 1.1 from vx_convolutional_network_rounding_type_e to vx_nn_rounding_type_e
*/
typedef    enum vx_nn_rounding_type_e                                vx_convolutional_network_rounding_type_e;
#define    VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR           VX_NN_DS_SIZE_ROUNDING_FLOOR
#define    VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING         VX_NN_DS_SIZE_ROUNDING_CEILING

/*
  keep the backward compatibility with spec 1.1 from vx_convolutional_network_pooling_type_e to vx_nn_pooling_type_e
*/
typedef    enum vx_nn_pooling_type_e                                 vx_convolutional_network_pooling_type_e;
#define    VX_CONVOLUTIONAL_NETWORK_POOLING_MAX                      VX_NN_POOLING_MAX
#define    VX_CONVOLUTIONAL_NETWORK_POOLING_AVG                      VX_NN_POOLING_AVG
#define    VX_CONVOLUTIONAL_NETWORK_POOLING_L2                       VX_NN_POOLING_L2
#define    VX_CONVOLUTIONAL_NETWORK_POOLING_AVG_ANDROID              VX_NN_POOLING_AVG_ANDROID

/*
  keep the backward compatibility with spec 1.1 from vx_convolutional_network_norm_type_e to vx_nn_norm_type_e
*/
typedef    enum vx_nn_norm_type_e                        vx_convolutional_network_norm_type_e;
#define    VX_CONVOLUTIONAL_NETWORK_NORM_SAME_MAP        VX_NN_NORMALIZATION_SAME_MAP
#define    VX_CONVOLUTIONAL_NETWORK_NORM_ACROSS_MAPS     VX_NN_NORMALIZATION_ACROSS_MAPS

/*
  keep the backward compatibility with spec 1.1 from vx_convolutional_network_layer_type_e to vx_nn_layer_type_e
*/
typedef    enum vx_nn_layer_type_e                           vx_convolutional_network_layer_type_e;
#define    VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER        VX_NN_CONVOLUTION_LAYER
#define    VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER     VX_NN_FULLYCONNECTED_LAYER

/*
  keep the backward compatibility with spec 1.1 from vx_convolutional_network_activation_func_e to
  vx_nn_activation_function_e
*/
typedef    enum vx_nn_activation_function_e                                vx_convolutional_network_activation_func_e;
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LOGISTIC                    VX_NN_ACTIVATION_LOGISTIC
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HYPERBOLIC_TAN              VX_NN_ACTIVATION_HYPERBOLIC_TAN
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RELU                        VX_NN_ACTIVATION_RELU
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_BRELU                       VX_NN_ACTIVATION_BRELU
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SOFTRELU                    VX_NN_ACTIVATION_SOFTRELU
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ABS                         VX_NN_ACTIVATION_ABS
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SQUARE                      VX_NN_ACTIVATION_SQUARE
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SQRT                        VX_NN_ACTIVATION_SQRT
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LINEAR                      VX_NN_ACTIVATION_LINEAR
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LEAKYRELU                   VX_NN_ACTIVATION_LEAKYRELU
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RELU6                       VX_NN_ACTIVATION_RELU6
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RELU1                       VX_NN_ACTIVATION_RELU1
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RSQRT                       VX_NN_ACTIVATION_RSQRT
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LEAKYRELU_MAX_POOLING       VX_NN_ACTIVATION_LEAKYRELU_MAX_POOLING
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_NONE                        VX_NN_ACTIVATION_NONE
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SWISH                       VX_NN_ACTIVATION_SWISH
#define    VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HWISH                       VX_NN_ACTIVATION_HSWISH

/*
  keep the backward compatibility with spec 1.1 for vxCopyTensorPatch_11
*/
VX_API_ENTRY vx_status VX_API_CALL
vxCopyTensorPatch_11(
    vx_tensor         tensor,
    vx_tensor_view    view,
    vx_tensor_addressing    user_addr,
    void              *user_ptr,
    vx_enum           usage,
    vx_enum           user_mem_type
);
//#define vxCopyTensorPatch         vxCopyTensorPatch_11

#if defined(__cplusplus)
}
#endif
#endif
