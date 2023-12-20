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
#ifndef TIM_VX_TENSOR_H_
#define TIM_VX_TENSOR_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "tim/vx/types.h"

namespace tim {
namespace vx {

using ShapeType = std::vector<uint32_t>;

class Quantization {
 public:
  Quantization() : type_(QuantType::NONE) {}
  Quantization(QuantType type, float scale, int32_t zero_point)
      : type_(type), scales_({scale}), zero_points_({zero_point}) {}
  Quantization(QuantType type, int32_t channel_dim, std::vector<float> scales,
               std::vector<int32_t> zero_points)
      : type_(type),
        channel_dim_(channel_dim),
        scales_(std::move(scales)),
        zero_points_(std::move(zero_points)) {}
  Quantization(QuantType type, int8_t fl) : type_(type), fl_(fl) {}
  QuantType& Type() { return type_; }
  const QuantType& Type() const { return type_; }
  Quantization& SetType(QuantType type) {
    this->type_ = type;
    return *this;
  }

  int32_t& ChannelDim() { return this->channel_dim_; }
  const int32_t& ChannelDim() const { return this->channel_dim_; }
  Quantization& SetChannelDim(int32_t channel_dim) {
    this->channel_dim_ = channel_dim;
    return *this;
  }

  std::vector<float>& Scales() { return this->scales_; }
  const std::vector<float>& Scales() const { return this->scales_; }
  Quantization& SetScales(std::vector<float> scales) {
    this->scales_ = scales;
    return *this;
  }

  std::vector<int32_t>& ZeroPoints() { return this->zero_points_; }
  const std::vector<int32_t>& ZeroPoints() const { return this->zero_points_; }
  Quantization& SetZeroPoints(std::vector<int32_t> zero_points) {
    this->zero_points_ = zero_points;
    return *this;
  }

  const std::int8_t& Fl() const { return this->fl_; }

  bool operator==(const Quantization& other_quant) const;

 protected:
  QuantType type_{QuantType::NONE};
  int32_t channel_dim_{-1};
  std::vector<float> scales_;
  std::vector<int32_t> zero_points_;
  int8_t fl_ = 0;
};

struct TensorSpec {
  TensorSpec() {}
  TensorSpec(DataType datatype, const ShapeType& shape, TensorAttribute attr)
      : datatype_(datatype), shape_(shape), attr_(attr) {}

  TensorSpec(DataType datatype, const ShapeType& shape, TensorAttribute attr,
             const Quantization& quantization)
      : TensorSpec(datatype, shape, attr) {
    this->quantization_ = quantization;
  }

  TensorSpec(const TensorSpec& other);

  bool operator==(const TensorSpec& other_spec) const;

  TensorSpec& operator=(const TensorSpec& other);

  TensorSpec& SetDataType(DataType datatype);

  TensorSpec& SetShape(const ShapeType& shape);

  TensorSpec& SetAttribute(TensorAttribute attr);

  TensorSpec& SetQuantization(Quantization& quantization);

  TensorSpec AsTransientSpec() const;

  int64_t GetElementNum() const;

  int64_t GetElementByteSize() const;

  int64_t GetByteSize() const;

  inline DataType& GetDataType() { return datatype_; }

  inline ShapeType& GetShapeType() { return shape_; }

  inline TensorAttribute& GetTensorAttribute() { return attr_; }

  inline Quantization& GetQuantization() { return quantization_; }

  DataType datatype_;
  ShapeType shape_;
  TensorAttribute attr_;
  Quantization quantization_;
};

struct DmaBufferDesc {
  int64_t fd;
};

class Tensor {
 public:
  virtual ~Tensor() {}
  virtual const ShapeType& GetShape() = 0;
  virtual DataType GetDataType() = 0;
  virtual const Quantization& GetQuantization() = 0;
  virtual TensorSpec& GetSpec() = 0;
  virtual uint32_t GetId() = 0;
  virtual bool CopyDataToTensor(const void* data,
                                uint32_t size_in_bytes = 0) = 0;
  virtual bool CopyDataFromTensor(void* data) = 0;
  virtual bool SwapHandle(void* new_ptr, bool is_new_ptr_malloc_by_ovxlib,
                          void** old_ptr) = 0;
  virtual bool SwapHandle(std::shared_ptr<tim::vx::Tensor> tensor) = 0;
  virtual bool SwapHandleWithCache(std::shared_ptr<tim::vx::Tensor> tensor) = 0;
  virtual bool FlushCacheForHandle() = 0;
  virtual bool InvalidateCacheForHandle() = 0;
  virtual void* map(bool invalidate_cpu_cache = false) = 0;
  virtual void unmap() = 0;
  virtual bool IsPlaceHolder() = 0;
  virtual bool IsConstTensor() = 0;
  virtual bool IsScalar() = 0;
  virtual bool SaveTensorToTextByFp32(std::string filename) = 0;
  virtual void SetScalar(int8_t is_scalar) = 0;
  virtual void* ConvertTensorToData(uint8_t* tensorData) = 0;
  virtual float* ConvertTensorToFloat32Data() = 0;
};
namespace utils {
bool Float32ToDtype(std::shared_ptr<tim::vx::Tensor> tensor,
                    std::vector<float> fval, uint8_t* tensorData);
bool DtypeToFloat32(std::shared_ptr<tim::vx::Tensor> tensor,
                    uint8_t* tensorData, float* data);
}  //namespace utils
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_TENSOR_H_ */
