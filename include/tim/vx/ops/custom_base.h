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
#ifndef TIM_VX_OPS_CUSTOM_BASE_H_
#define TIM_VX_OPS_CUSTOM_BASE_H_

#include "tim/vx/builtin_op.h"
#include <typeinfo>
#include <tuple>

namespace tim {
namespace vx {
namespace ops {
#define gpu_align(n, align) ((n) + ((align)-1)) & ~((align)-1)

__attribute__((unused)) static int32_t gobal_kernel_id_ = 0;

struct Param {
  union {
    float f;
    int i;
    bool b;
    uint ui;
  } data;
  tim::vx::DataType type;
};

template <size_t I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type param_transform(
    std::tuple<Ts...> tup, std::vector<Param>& param_list) {
  auto size = std::tuple_size<decltype(tup)>::value;
  if (size == I && param_list.size() != 0) {
    //nothing to do
  }
  return;
}

template <size_t I = 0, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type param_transform(
    std::tuple<Ts...> tup, std::vector<Param>& param_list) {
  if (typeid(std::get<I>(tup)) == typeid(float)) {
    Param p;
    p.data.f = std::get<I>(tup);
    p.type = tim::vx::DataType::FLOAT32;
    param_list.push_back(p);
  } else if (typeid(std::get<I>(tup)) == typeid(uint32_t)) {
    Param p;
    p.data.ui = std::get<I>(tup);
    p.type = tim::vx::DataType::UINT32;
    param_list.push_back(p);
  } else if (typeid(std::get<I>(tup)) == typeid(int32_t)) {
    Param p;
    p.data.i = std::get<I>(tup);
    p.type = tim::vx::DataType::INT32;
    param_list.push_back(p);
  } else if (typeid(std::get<I>(tup)) == typeid(bool)) {
    Param p;
    p.data.b = std::get<I>(tup);
    p.type = tim::vx::DataType::BOOL8;
    param_list.push_back(p);
  } else {
    std::cout << "need more else if type of param list " << std::endl;
  }
  // Go to next element
  param_transform<I + 1>(tup, param_list);
}

class CustomOpBase : public Operation {
 public:
  CustomOpBase(Graph* graph, uint32_t input_num, uint32_t output_num,
               int32_t kernel_id, const char* kernel_name);

  ~CustomOpBase();
  virtual void SetupParams(
      std::vector<tim::vx::DataType> input_types,
      std::string& build_option) = 0;

  virtual void SetupShapeInfor() = 0;

  virtual void SetupEnqueue(uint32_t& dim, std::vector<size_t>& gobal_size,
                            std::vector<size_t>& local_size) = 0;

  std::vector<Param> param_list_;
  std::vector<tim::vx::ShapeType> inputs_size_;
  std::vector<tim::vx::ShapeType> outputs_size_;

  const char* func_name_;
  const char* kernel_resource_;
  void* init_kernel_;
  void* vx_node_;

  uint32_t input_num_;
  uint32_t output_num_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_CUSTOM_BASE_H_ */