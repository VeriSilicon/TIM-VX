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
#ifndef TIM_VX_TENSOR_H_
#define TIM_VX_TENSOR_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

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

  QuantType Type() { return type_; }
  Quantization& SetType(QuantType type) {
    this->type_ = type;
    return *this;
  }

  int32_t ChannelDim() { return this->channel_dim_; }
  Quantization& SetChannelDim(int32_t channel_dim) {
    this->channel_dim_ = channel_dim;
    return *this;
  }

  std::vector<float>& Scales() { return this->scales_; }
  Quantization& SetScales(std::vector<float> scales) {
    this->scales_ = scales;
    return *this;
  }

  std::vector<int32_t>& ZeroPoints() { return this->zero_points_; }
  Quantization& SetZeroPoints(std::vector<int32_t> zero_points) {
    this->zero_points_ = zero_points;
    return *this;
  }

 protected:
  QuantType type_{QuantType::NONE};
  int32_t channel_dim_{-1};
  std::vector<float> scales_;
  std::vector<int32_t> zero_points_;
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

  TensorSpec(const TensorSpec& other) {
	  this->datatype_ = other.datatype_;
	  this->shape_ = other.shape_;
	  this->attr_ = other.attr_;
	  this->quantization_  = other.quantization_;
  }

  TensorSpec& operator =(const TensorSpec& other) {
    this->datatype_ = other.datatype_;
	  this->shape_ = other.shape_;
	  this->attr_ = other.attr_;
	  this->quantization_  = other.quantization_;
    return *this;
  }

  TensorSpec& SetDataType(DataType datatype) {
    this->datatype_ = datatype;
    return *this;
  }

  TensorSpec& SetShape(ShapeType& shape) {
    this->shape_ = shape;
    return *this;
  }

  TensorSpec& SetAttribute(TensorAttribute attr) {
    this->attr_ = attr;
    return *this;
  }

  TensorSpec& SetQuantization(Quantization& quantization) {
    this->quantization_ = quantization;
    return *this;
  }

  TensorSpec AsTransientSpec() const {
    return TensorSpec(this->datatype_, ShapeType({}),
                      TensorAttribute::TRANSIENT, this->quantization_);
  }

  DataType datatype_;
  ShapeType shape_;
  TensorAttribute attr_;
  Quantization quantization_;
};

class Tensor {
 public:
  virtual ~Tensor() {}
  virtual const ShapeType& GetShape() = 0;
  virtual DataType GetDataType() = 0;
  virtual const Quantization& GetQuantization() = 0;
  virtual TensorSpec& GetSpec() = 0;
  virtual uint32_t GetId() = 0;
  virtual bool CopyDataToTensor(const void* data, uint32_t size_in_bytes = 0) = 0;
  virtual bool CopyDataFromTensor(void* data) = 0;
  virtual bool IsPlaceHolder() = 0;
  virtual bool IsConstTensor() = 0;
  virtual const void* GetDataRef() const = 0;
};

}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_TENSOR_H_ */
