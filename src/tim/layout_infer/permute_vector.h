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
#ifndef TIM_VX_LAYOUT_INFERENCE_PERMUTE_VECTOR_H_
#define TIM_VX_LAYOUT_INFERENCE_PERMUTE_VECTOR_H_

#include <array>
#include <cassert>
#include <memory>
#include <vector>
#include <string>

namespace tim {
namespace transform {
class IPermuteVector;
using IPermuteVectorPtr = std::shared_ptr<IPermuteVector>;

class IPermuteVector {
 public:
  virtual ~IPermuteVector() = default;
  virtual uint32_t Rank() const = 0;

  virtual const uint32_t& At(const uint32_t) const = 0;
  virtual uint32_t& At(const uint32_t) = 0;

  /**
     * @brief get Reverse permute vector
     *
     * PermuteVector + PermuteVector.Reverse() = {0, 1, 2...R}
     *
     * Data layout = NHWC, current Permute = 0, 3, 1, 2, output layout = NCHW
     * its reverse layout is 0, 2, 3, 1
     *
     * @return PermuteVector<R> reverse permute vector have same rank as current permute
     */
  virtual IPermuteVectorPtr Reverse() = 0;

  virtual std::string AsText() const = 0;

  /**
     * @brief apply addtional permute parameter
     *
     * @detail
     *   assume data stored as NHWC, this->param_ = {0, 3, 1, 2}
     *   if apply current permute vector, data stored as NCHW
     *   other->param_ = {0, 2, 1, 3}
     *   if apply the addtion permute, data stored as NHCW, current permute paramter become {0, 1,
     * 3, 2}
     *
     * @param other addtional permute vector
     * @return PermuteVector result = data.apply_this_permute().apply_other_permute()
     */
  virtual IPermuteVectorPtr Add(const IPermuteVectorPtr& other) const = 0;

  virtual void ReInitialize() = 0;

  virtual bool IsAligned() const = 0;

  virtual std::vector<uint32_t> AsStdVec() const = 0;
};

template <uint32_t R>
class PermuteVector : public IPermuteVector {
 public:
  static constexpr uint32_t kMaxRank = 10;

  PermuteVector() {
    for (uint32_t i = 0; i < R; ++i) {
      param_[i] = i;
    }
  }
  // Copy Constructor
  PermuteVector(const PermuteVector& other) : param_(other.param_) {}
  PermuteVector& operator=(const PermuteVector& other) {
    assert(this != &other);
    this->param_ = other.param_;
    return *this;
  }

  // Move Constructor
  PermuteVector(PermuteVector&& other) : param_(std::move(other.param_)) {}
  PermuteVector& operator=(PermuteVector&& other) {
    assert(this != &other);
    this->param_ = std::move(other.param_);
    return *this;
  }

  // Initialize list
  PermuteVector(std::initializer_list<uint32_t> init_list) {
    std::vector<uint32_t> vec(init_list);
    assert(vec.size() == R);

    for (uint32_t i = 0; i < R; ++i) {
      param_[i] = vec[i];
    }
  }

  template <uint32_t S>
  explicit PermuteVector(const PermuteVector<S>& smaller) {
    // With this: you can construct a PermuteVector with larger Rank from a smaller rank permute
    static_assert(S < R, "Cut Permute Vector is not allowed");
    for (auto i = 0; i < R; ++i) {
      param_[i] = i < S ? smaller[i] : i;
    }
  }

  const uint32_t& At(uint32_t idx) const override { return param_[idx]; }

  uint32_t& At(uint32_t idx) override { return param_[idx]; }

  uint32_t Rank() const override { return R; }

  bool IsAligned() const override {
    uint32_t i = 0;
    for (; i < R; ++i) {
      if (i != param_[i]) break;
    }

    return i == R;
  }

  IPermuteVectorPtr Reverse() override {
    IPermuteVectorPtr r = std::make_shared<PermuteVector<R>>();
    for (uint32_t i = 0; i < R; ++i) {
      r->At(param_[i]) = i;
    }
    return r;
  }

  void ReInitialize() override {
    for (uint32_t i = 0; i < R; ++i) {
      param_[i] = i;
    }
  }

  IPermuteVectorPtr Add(const IPermuteVectorPtr& other) const override {
    IPermuteVectorPtr r = std::make_shared<PermuteVector<R>>();
    for (uint32_t i = 0; i < other->Rank(); ++i) {
      r->At(i) = param_[other->At(i)];
    }
    return r;
  }

  virtual std::string AsText() const override {
    std::string str(R + 1, '\0');
    for (uint32_t i = 0; i < R; i++) {
      str[i] = (char(param_[i]));
    }
    return str;
  }

  virtual std::vector<uint32_t> AsStdVec() const override {
    std::vector<uint32_t> data(R);

    for (uint32_t i(0); i < R; ++i) {
      data[i] = param_[i];
    }
    return data;
  }

 private:
  std::array<uint32_t, R> param_;
};

/**
 * @brief
 *
 * @param rank_val
 * @return IPermuteVectorPtr
 */
inline IPermuteVectorPtr MakeShared(uint32_t rank_val) {
  switch (rank_val) {
    // 0: represent scalar
    case 0:
    case 1:
      return std::make_shared<PermuteVector<1>>();
    case 2:
      return std::make_shared<PermuteVector<2>>();
    case 3:
      return std::make_shared<PermuteVector<3>>();
    case 4:
      return std::make_shared<PermuteVector<4>>();
    case 5:
      return std::make_shared<PermuteVector<5>>();
    case 6:
      return std::make_shared<PermuteVector<6>>();
    case 7:
      return std::make_shared<PermuteVector<7>>();
    case 8:
      return std::make_shared<PermuteVector<8>>();
    case 9:
      return std::make_shared<PermuteVector<9>>();
    case 10:
      return std::make_shared<PermuteVector<10>>();
    default:
      assert("Not supported rankVal");
      return nullptr;
  }
}

}  // namespace transform
}  // namespace tim

#endif
