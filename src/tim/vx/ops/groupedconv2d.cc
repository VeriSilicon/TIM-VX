/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/groupedconv2d.h"

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

GroupedConv2d::GroupedConv2d(Graph* graph,
               PadType padding,
               const std::array<uint32_t, 2>& strides,
               const std::array<uint32_t, 2>& dilation,
               int32_t group_number,
               DataLayout input_layout, DataLayout kernel_layout)
    : Operation(graph, VSI_NN_OP_GROUPED_CONV2D, 3, 1, input_layout),
      padding_(padding), strides_(strides), dilation_(dilation),
      pad_({0,0,0,0}), group_number_(group_number),
      kernel_layout_(kernel_layout) {
  this->impl()->node()->nn_param.conv2d.stride[0] = strides_[0];
  this->impl()->node()->nn_param.conv2d.stride[1] = strides_[1];
  this->impl()->node()->nn_param.conv2d.pad_type = TranslatePadType(padding_);
  this->impl()->node()->nn_param.conv2d.group = group_number_;
  this->impl()->node()->nn_param.conv2d.dilation[0] = dilation_[0];
  this->impl()->node()->nn_param.conv2d.dilation[1] = dilation_[1];
  }

GroupedConv2d::GroupedConv2d(Graph* graph,
               const std::array<uint32_t, 4>& pad,
               const std::array<uint32_t, 2>& strides,
               const std::array<uint32_t, 2>& dilation,
               int32_t group_number,
               DataLayout input_layout, DataLayout kernel_layout)
    : Operation(graph, VSI_NN_OP_GROUPED_CONV2D, 3, 1, input_layout),
      padding_(PadType::AUTO), strides_(strides), dilation_(dilation), pad_(pad),
      group_number_(group_number), kernel_layout_(kernel_layout) {
  this->impl()->node()->nn_param.conv2d.stride[0] = strides_[0];
  this->impl()->node()->nn_param.conv2d.stride[1] = strides_[1];
  this->impl()->node()->nn_param.conv2d.group = group_number_;
  this->impl()->node()->nn_param.conv2d.dilation[0] = dilation_[0];
  this->impl()->node()->nn_param.conv2d.dilation[1] = dilation_[1];
}

std::shared_ptr<Operation> GroupedConv2d::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<GroupedConv2d>(
      this->pad_, this->strides_, this->dilation_, this->group_number_,
      this->impl_->layout_, this->kernel_layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim