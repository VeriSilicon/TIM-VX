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
#ifndef _NPU_REFERENCE_INTERFACE_H
#define _NPU_REFERENCE_INTERFACE_H
/** npu reference for quantized conv2d
 * pad      [pad top, pad bottom, pad left, pad right]
 * strides  [stride height, stride width]
 * dilation [dilation height, dilation width]
 */
void npuref_interface_quant_conv2d(
    const void* input_buffer, const vsi_nn_kernel_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_kernel_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_kernel_tensor_attr_t* output_attr, void* output_buffer);

/** npu reference for quant deconv2d
 * pad      [pad top, pad bottom, pad left, pad right]
 * strides  [stride height, stride width]
 * dilation [dilation height, dilation width]
 */
void npuref_interface_quant_deconv2d(
    const void* input_buffer, const vsi_nn_kernel_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_kernel_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_kernel_tensor_attr_t* output_attr, void* output_buffer);


/** npu reference for quant depthwise conv2d
 */
void npuref_interface_quant_depthwise_conv2d(
    const void* input_buffer,
    const void* kernel_buffer,
    const void* bias_buffer,
    const int32_t* input_shape, uint32_t input_dim,
    const int32_t* kernel_shape, uint32_t kernel_dim,
    const int32_t* output_shape, uint32_t output_dim,
    float input_scale, int32_t input_zero_point,
    float kernel_scale, int32_t kernel_zero_point,
    float output_scale, int32_t output_zero_point,
    int32_t pad_h_front, int32_t pad_h_end,
    int32_t pad_w_front, int32_t pad_w_end,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    void* output_buffer);

vsi_bool npuref_exists();

void npuref_init();

void npuref_shutdown();

#endif
