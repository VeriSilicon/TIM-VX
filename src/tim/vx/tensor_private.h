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
  TensorImpl(Graph* graph, const TensorSpec& spec, const DmaBufferDesc& dmafd);
  TensorImpl(Graph* graph, const TensorSpec& spec, void* data = nullptr);
  ~TensorImpl();

  bool Init(void* external_cache = nullptr);
  bool IsWriteable();
  bool IsReadable();

  const ShapeType& GetShape() override { return spec_.shape_; }
  DataType GetDataType() override { return spec_.datatype_; }
  const Quantization& GetQuantization() override { return spec_.quantization_; }
  TensorSpec& GetSpec() override { return spec_; }
  uint32_t GetId() override;
  bool CopyDataToTensor(const void* data, uint32_t size = 0) override;
  bool CopyDataFromTensor(void* data) override;
  bool SwapHandle(void* new_ptr, bool is_new_ptr_malloc_by_ovxlib,
                  void** old_ptr) override;
  bool SwapHandle(std::shared_ptr<tim::vx::Tensor> tensor) override;
  bool SwapHandleWithCache(std::shared_ptr<tim::vx::Tensor> tensor) override;
  bool FlushCacheForHandle() override;
  bool InvalidateCacheForHandle() override;
  void* map(bool invalidate_cpu_cache = false) override;
  void unmap() override;
  bool IsPlaceHolder() override { return false; }
  bool IsConstTensor() override {
    return spec_.attr_ == tim::vx::TensorAttribute::CONSTANT;
  }
  bool IsScalar() override {
    return vsi_nn_GetTensorIsScalar(vsi_nn_GetTensor(graph_->graph(), id_));
  }
  bool SaveTensorToTextByFp32(std::string filename) override;
  void* ConvertTensorToData(uint8_t* tensorData) override;
  float* ConvertTensorToFloat32Data() override;
  void SetScalar(int8_t is_scalar) override;

  GraphImpl* graph_;
  vsi_nn_tensor_id_t id_;
  TensorSpec spec_;
  void* data_;
  int64_t fd_{-1};
};

class TensorPlaceholder : public Tensor {
 public:
  TensorPlaceholder(Graph* graph) : id_(VSI_NN_TENSOR_ID_NA) { (void)(graph); }
  ~TensorPlaceholder(){};

  const ShapeType& GetShape() override { return spec_.shape_; }
  DataType GetDataType() override { return spec_.datatype_; }
  const Quantization& GetQuantization() override { return spec_.quantization_; }
  TensorSpec& GetSpec() override { return spec_; }
  uint32_t GetId() override { return id_; };
  bool CopyDataToTensor(const void* data, uint32_t size = 0) override {
    (void)data, void(size);
    return false;
  }
  bool CopyDataFromTensor(void* data) override {
    (void)data;
    return false;
  }
  bool SwapHandle(void* new_ptr, bool is_new_ptr_malloc_by_ovxlib,
                  void** old_ptr) override {
    (void)new_ptr;
    (void)old_ptr;
    (void)is_new_ptr_malloc_by_ovxlib;
    return false;
  }
  bool SwapHandle(std::shared_ptr<tim::vx::Tensor> tensor) override {
    (void)tensor;
    return false;
  }
  bool SwapHandleWithCache(std::shared_ptr<tim::vx::Tensor> tensor) override {
    (void)tensor;
    return false;
  }
  bool InvalidateCacheForHandle() override { return false; }
  bool FlushCacheForHandle() override { return false; }
  void* map(bool invalidate_cpu_cache = false) override {
    (void)invalidate_cpu_cache;
    return nullptr;
  }
  void unmap() override { return; }
  bool IsPlaceHolder() override { return true; }
  bool IsConstTensor() override {
    return spec_.attr_ == tim::vx::TensorAttribute::CONSTANT;
  }
  bool IsScalar() override {
    return false;
  }
  bool SaveTensorToTextByFp32(std::string filename) override {
    (void)filename;
    return false;
  }
  void* ConvertTensorToData(uint8_t* tensorData) override {
    (void)tensorData;
    return nullptr;
  }
  float* ConvertTensorToFloat32Data() override { return nullptr; }

  void SetScalar(int8_t is_scalar) override { (void) is_scalar; return; }
  vsi_nn_tensor_id_t id_;
  TensorSpec spec_;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_TENSOR_PRIVATE_H_ */