/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
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
#include "tim/vx/ops/conv3d.h"
#include "builtin_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
Conv3d::Conv3d(Graph* graph, PadType padding,
               const std::array<int32_t, 3>& stride,
               const std::array<int32_t, 3>& dilation, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : Conv3d(graph, 0, padding, {0, 0, 0}, stride, dilation, {0, 0, 0, 0, 0, 0},
             multiplier, input_layout, kernel_layout) {}

Conv3d::Conv3d(Graph* graph, const std::array<int32_t, 6> pad,
               const std::array<int32_t, 3>& stride,
               const std::array<int32_t, 3>& dilation, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : Conv3d(graph, 0, PadType::AUTO, {0, 0, 0}, stride, dilation, pad,
             multiplier, input_layout, kernel_layout) {}

Conv3d::Conv3d(Graph* graph, int32_t weights, PadType padding,
               const std::array<int32_t, 3>& ksize,
               const std::array<int32_t, 3>& stride,
               const std::array<int32_t, 3>& dilation, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : Conv3d(graph, weights, padding, ksize, stride, dilation,
             {0, 0, 0, 0, 0, 0}, multiplier, input_layout, kernel_layout) {}

Conv3d::Conv3d(Graph* graph, int32_t weights, PadType padding,
               const std::array<int32_t, 3>& ksize,
               const std::array<int32_t, 3>& stride,
               const std::array<int32_t, 3>& dilation,
               const std::array<int32_t, 6>& pad, int32_t multiplier,
               DataLayout input_layout, DataLayout kernel_layout)
    : BuiltinOp(graph, VSI_NN_OP_CONV3D, 0, 0, input_layout),
      weights_(weights),
      padding_(padding),
      ksize_(ksize),
      stride_(stride),
      dilation_(dilation),
      pad_(pad),
      multiplier_(multiplier),
      kernel_layout_(kernel_layout) {
  this->impl()->node()->nn_param.conv3d.stride[0] = stride_[0];
  this->impl()->node()->nn_param.conv3d.stride[1] = stride_[1];
  this->impl()->node()->nn_param.conv3d.stride[2] = stride_[2];
  this->impl()->node()->nn_param.conv3d.pad_type = TranslatePadType(padding_);
  this->impl()->node()->nn_param.conv3d.dilation[0] = dilation_[0];
  this->impl()->node()->nn_param.conv3d.dilation[1] = dilation_[1];
  this->impl()->node()->nn_param.conv3d.dilation[2] = dilation_[2];
  this->impl()->node()->nn_param.conv3d.pad[0] = pad_[0];
  this->impl()->node()->nn_param.conv3d.pad[1] = pad_[1];
  this->impl()->node()->nn_param.conv3d.pad[2] = pad_[2];
  this->impl()->node()->nn_param.conv3d.pad[3] = pad_[3];
  this->impl()->node()->nn_param.conv3d.pad[4] = pad_[4];
  this->impl()->node()->nn_param.conv3d.pad[5] = pad_[5];
  this->impl()->node()->nn_param.conv3d.weights = weights_;
  this->impl()->node()->nn_param.conv3d.multiplier = multiplier_;
}

std::shared_ptr<Operation> Conv3d::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Conv3d>(
      this->weights_, this->padding_, this->ksize_, this->stride_,
      this->dilation_, this->pad_, this->multiplier_, this->impl_->layout_,
      this->kernel_layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim