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
#include "tim/vx/ops/groupedconv1d.h"

#include "builtin_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

GroupedConv1d::GroupedConv1d(Graph* graph, PadType padding,
                             const uint32_t stride, const uint32_t dilation,
                             uint32_t group, DataLayout input_layout,
                             DataLayout kernel_layout)
    : GroupedConv1d(graph, padding, {0, 0}, stride, dilation, group, input_layout, kernel_layout) {}

GroupedConv1d::GroupedConv1d(Graph* graph, PadType padding,
                             std::array<uint32_t, 2> pad, const uint32_t stride,
                             const uint32_t dilation, uint32_t group,
                             DataLayout input_layout, DataLayout kernel_layout)
    : BuiltinOp(graph, VSI_NN_OP_GROUPED_CONV1D, 3, 1, input_layout),
      padding_(padding),
      pad_(pad),
      stride_(stride),
      dilation_(dilation),
      group_(group),
      kernel_layout_(kernel_layout) {
  this->impl()->node()->nn_param.grouped_conv1d.pad_type =
      TranslatePadType(padding_);
  this->impl()->node()->nn_param.grouped_conv1d.pad[0] = pad_[0];
  this->impl()->node()->nn_param.grouped_conv1d.pad[1] = pad_[1];
  this->impl()->node()->nn_param.grouped_conv1d.stride = stride_;
  this->impl()->node()->nn_param.grouped_conv1d.group = group_;
  this->impl()->node()->nn_param.grouped_conv1d.dilation = dilation_;
}

std::shared_ptr<Operation> GroupedConv1d::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<GroupedConv1d>(
      this->padding_, this->pad_, this->stride_, this->dilation_, this->group_,
      this->impl_->layout_, this->kernel_layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim