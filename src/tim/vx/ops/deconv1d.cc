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
#include "tim/vx/ops/deconv1d.h"

#include <cassert>

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

DeConv1d::DeConv1d(Graph* graph, PadType pad_type,
    uint32_t stride, uint32_t output_padding, uint32_t group,
    DataLayout input_layout, DataLayout kernel_layout)
  : DeConv1d(graph, pad_type, stride, output_padding, {0, 0}, group,
    input_layout, kernel_layout) {
}

DeConv1d::DeConv1d(Graph* graph, const std::array<uint32_t, 2>& pad,
    uint32_t stride, uint32_t output_padding, uint32_t group,
    DataLayout input_layout, DataLayout kernel_layout)
  : DeConv1d(graph, PadType::AUTO, stride, output_padding, pad, group,
    input_layout, kernel_layout) {
}

DeConv1d::DeConv1d(Graph* graph, int32_t oc_count, PadType pad_type,
    uint32_t ksize, uint32_t stride, uint32_t output_padding)
  : DeConv1d(graph, pad_type, stride, output_padding,
      {0, 0}, 1, DataLayout::WHCN, DataLayout::WHIcOc) {
  (void)ksize;
  (void)oc_count;
}

DeConv1d::DeConv1d(Graph* graph, int32_t oc_count, PadType pad_type,
    uint32_t ksize, uint32_t stride, uint32_t output_padding,
    const std::array<uint32_t, 2>& pad, uint32_t group)
  : DeConv1d(graph, pad_type, stride, output_padding,
    pad, group, DataLayout::WHCN, DataLayout::WHIcOc) {
  (void)ksize;
  (void)oc_count;
}

DeConv1d::DeConv1d(Graph* graph, PadType pad_type,
    uint32_t stride, uint32_t output_padding,
    const std::array<uint32_t, 2>& pad, uint32_t group,
    DataLayout input_layout, DataLayout kernel_layout)
  : Operation(graph, VSI_NN_OP_DECONVOLUTION1D, 3, 1, input_layout),
    oc_count_(0),
    pad_type_(pad_type),
    ksize_(0),
    stride_(stride),
    output_padding_(output_padding),
    pad_(pad),
    group_(group),
    kernel_layout_(kernel_layout) {
  this->impl()->node()->nn_param.deconvolution1d.stride = stride_;
  this->impl()->node()->nn_param.deconvolution1d.pad_type = TranslatePadType(pad_type_);
  this->impl()->node()->nn_param.deconvolution1d.group = group_;
  this->impl()->node()->nn_param.deconvolution1d.output_padding = output_padding_;
  this->impl()->node()->nn_param.deconvolution1d.pad[0] = pad_[0];
  this->impl()->node()->nn_param.deconvolution1d.pad[1] = pad_[1];
}

std::shared_ptr<Operation> DeConv1d::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<DeConv1d>(
      this->pad_type_, this->stride_, this->output_padding_, this->pad_,
      this->group_, this->impl_->layout_, this->kernel_layout_);
}
} // namespace ops
} // namespace vx
} // namespace tim
