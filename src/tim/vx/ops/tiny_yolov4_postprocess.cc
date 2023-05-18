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
#include "tim/vx/ops/tiny_yolov4_postprocess.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"
#ifdef VSI_FEAT_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS
namespace tim {
namespace vx {
namespace ops {
TinyYolov4Postprocess::TinyYolov4Postprocess(Graph* graph)
    : BuiltinOp(graph, VSI_NN_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS, 4, 2){}

std::shared_ptr<Operation> TinyYolov4Postprocess::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<TinyYolov4Postprocess>();
}
}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif //(FEAT_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS)