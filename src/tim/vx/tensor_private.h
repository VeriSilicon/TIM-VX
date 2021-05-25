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
#ifndef TIM_VX_TENSOR_PRIVATE_H_
#define TIM_VX_TENSOR_PRIVATE_H_
#include "graph_private.h"
#include "tim/vx/tensor.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {

class TensorImpl : public Tensor {
 public:
  TensorImpl(Graph* graph, const TensorSpec& spec, const void* data = nullptr);
  ~TensorImpl();

  bool Init();
  bool IsWriteable();
  bool IsReadable();

  const ShapeType& GetShape() { return spec_.shape_; }
  DataType GetDataType() { return spec_.datatype_; }
  const Quantization& GetQuantization() { return spec_.quantization_; }
  TensorSpec& GetSpec() { return spec_; }
  uint32_t GetId();
  bool CopyDataToTensor(const void* data, uint32_t size = 0);
  bool CopyDataFromTensor(void* data);
  bool IsPlaceHolder() { return false; }
  bool IsConstTensor() {
    return spec_.attr_ == tim::vx::TensorAttribute::CONSTANT;
  }
  const void* GetDataRef() const { return data_; }

  GraphImpl* graph_;
  vsi_nn_tensor_id_t id_;
  TensorSpec spec_;
  const void* data_;
};

class TensorPlaceholder : public Tensor {
 public:
  TensorPlaceholder(Graph* graph) : id_(VSI_NN_TENSOR_ID_NA) {(void)(graph);}
  ~TensorPlaceholder(){};

  const ShapeType& GetShape() { return spec_.shape_; }
  DataType GetDataType() { return spec_.datatype_; }
  const Quantization& GetQuantization() { return spec_.quantization_; }
  TensorSpec& GetSpec() { return spec_; }
  uint32_t GetId() { return id_; };
  bool CopyDataToTensor(const void* data, uint32_t size = 0) {
    (void)data, void(size);
    return false;
  }
  bool CopyDataFromTensor(void* data) {
    (void)data;
    return false;
  }
  bool IsPlaceHolder() { return true; }
  bool IsConstTensor() {
    return spec_.attr_ == tim::vx::TensorAttribute::CONSTANT;
  }
  const void* GetDataRef() const { return nullptr; }

  vsi_nn_tensor_id_t id_;
  TensorSpec spec_;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_TENSOR_PRIVATE_H_ */