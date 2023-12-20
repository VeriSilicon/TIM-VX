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
#ifndef TIM_EXPERIMENTAL_TRACE_TVX_TENSOR_H_
#define TIM_EXPERIMENTAL_TRACE_TVX_TENSOR_H_
#include "tim/vx/tensor.h"
#include "tim/experimental/trace/tvx/types.h"
#include "tim/experimental/trace/tracer.h"

namespace trace {

namespace target = ::tim::vx;

struct Quantization : public TraceClassBase<target::Quantization> {
  DEF_CONSTRUCTOR(Quantization)

  DEF_CONSTRUCTOR(Quantization, ((QuantType))
                                ((float))
                                ((int32_t))
  )

  DEF_CONSTRUCTOR(Quantization, ((QuantType))
                                ((int32_t))
                                ((std::vector<float>))
                                ((std::vector<int32_t>))
  )

  DEF_CONSTRUCTOR(Quantization, ((QuantType))
                                ((int8_t))
  )

  DEF_TRACED_API(QuantType&, Type)

  // DEF_TRACED_API(const QuantType&, Type)
  // lack of macro to def readonly apis
  template <class R = const QuantType &, class... Args>
  typename std::enable_if_t<is_not_traced_obj_like<R>::value, R> Type(
      Args... params) const {
    std::string this_obj_name = TraceGetObjName();
    Tracer::push_back_msg_cache(this_obj_name + "->" + __FUNCTION__ + "(");
    Tracer::clear_params_log_cache();
    boost::hana::tuple<Args...> params_tuple = {params...};
    boost::hana::for_each(params_tuple, [&](auto x) {
        Tracer::logging_param(x, -1);});
    Tracer::pop_params_log_cache();
    Tracer::amend_last_msg_cache(");\n");
    Tracer::msg_cache_sync_to_file();
    auto api_impl = [&](auto &&...ts) { return impl_->Type(ts...); };
    auto proc_param_lambda = [](auto &&t) { return Tracer::proc_param(t); };
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);
    return boost::hana::unpack(params_impl, api_impl);
  }

  DEF_TRACED_API(Quantization&, SetType)

  DEF_TRACED_API(int32_t&, ChannelDim)

  // DEF_TRACED_API(const int32_t&, ChannelDim)
  // lack of macro to def readonly apis
  template <class R = const int32_t &, class... Args>
  typename std::enable_if_t<is_not_traced_obj_like<R>::value, R> ChannelDim(
      Args... params) const {
    std::string this_obj_name = TraceGetObjName();
    Tracer::push_back_msg_cache(this_obj_name + "->" + __FUNCTION__ + "(");
    Tracer::clear_params_log_cache();
    boost::hana::tuple<Args...> params_tuple = {params...};
    boost::hana::for_each(params_tuple, [&](auto x) {
      Tracer::logging_param(x, -1); });
    Tracer::pop_params_log_cache();
    Tracer::amend_last_msg_cache(");\n");
    Tracer::msg_cache_sync_to_file();
    auto api_impl = [&](auto &&...ts) { return impl_->ChannelDim(ts...); };
    auto proc_param_lambda = [](auto &&t) { return Tracer::proc_param(t); };
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);
    return boost::hana::unpack(params_impl, api_impl);
  }

  DEF_TRACED_API(Quantization&, SetChannelDim)

  DEF_TRACED_API(std::vector<float>&, Scales)

  DEF_TRACED_API(Quantization&, SetScales)

  DEF_TRACED_API(std::vector<int32_t>&, ZeroPoints)

  // DEF_TRACED_API(const std::vector<int32_t>&, ZeroPoints)
  // lack of macro to def readonly apis
  template <class R = const std::vector<int32_t> &, class... Args>
  typename std::enable_if_t<is_not_traced_obj_like<R>::value, R> ZeroPoints(
      Args... params) const {
    std::string this_obj_name = TraceGetObjName();
    Tracer::push_back_msg_cache(this_obj_name + "->" + __FUNCTION__ + "(");
    Tracer::clear_params_log_cache();
    boost::hana::tuple<Args...> params_tuple = {params...};
    boost::hana::for_each(params_tuple, [&](auto x) {
        Tracer::logging_param(x, -1); });
    Tracer::pop_params_log_cache();
    Tracer::amend_last_msg_cache(");\n");
    Tracer::msg_cache_sync_to_file();
    auto api_impl = [&](auto &&...ts) { return impl_->ZeroPoints(ts...); };
    auto proc_param_lambda = [](auto &&t) { return Tracer::proc_param(t); };
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);
    return boost::hana::unpack(params_impl, api_impl);
  }

  DEF_TRACED_API(Quantization&, SetZeroPoints)

  DEF_TRACED_API(const std::int8_t&, Fl)
};

struct TensorSpec : public TraceClassBase<target::TensorSpec> {
  DEF_CONSTRUCTOR(TensorSpec)

  DEF_CONSTRUCTOR(TensorSpec, ((DataType))
                              ((const ShapeType&))
                              ((TensorAttribute))
  )

  DEF_CONSTRUCTOR(TensorSpec, ((DataType))
                              ((const ShapeType&))
                              ((TensorAttribute))
                              ((const Quantization&))
  )

  // DEF_CONSTRUCTOR(TensorSpec, ((const TensorSpec&))
  // )

  DEF_TRACED_API(TensorSpec&, operator=)

  DEF_TRACED_API(TensorSpec&, SetDataType)

  DEF_TRACED_API(TensorSpec&, SetShape)

  DEF_TRACED_API(TensorSpec&, SetAttribute)

  DEF_TRACED_API(TensorSpec&, SetQuantization)

  DEF_TRACED_API(TensorSpec&, AsTransientSpec)

  DEF_TRACED_API(int64_t, GetElementNum)

  DEF_TRACED_API(int64_t, GetElementByteSize)

  DEF_TRACED_API(int64_t, GetByteSize)

  DEF_SIMPLE_UNTRACED_API(DataType&, GetDataType)

  DEF_SIMPLE_UNTRACED_API(ShapeType&, GetShapeType)

  DEF_SIMPLE_UNTRACED_API(TensorAttribute&, GetTensorAttribute)

  DEF_TRACED_API(Quantization&, GetQuantization)
};

struct DmaBufferDesc : public TraceClassBase<target::DmaBufferDesc> {
  DEF_CONSTRUCTOR(DmaBufferDesc)
};

struct Tensor : public TraceClassBase<target::Tensor> {
  // DEF_INTERFACE_CONSTRUCTOR(Tensor)
  Tensor(const std::shared_ptr<target::Tensor> &impl) {
    Tracer::insert_traced_obj(static_cast<void *>(impl.get()),
        static_cast<void *>(this));
    impl_ = impl;
  }
  Tensor(const std::shared_ptr<target::Tensor> &impl, const TensorSpec& spec) {
    Tracer::insert_traced_obj(static_cast<void *>(impl.get()),
        static_cast<void *>(this));
    impl_ = impl;
    spec_ = spec;
  }
  // DEF_TRACED_API(const ShapeType&, GetShape)
  // unfixed issue of trace GetShape
  DEF_SIMPLE_UNTRACED_API(const ShapeType&, GetShape)

  DEF_TRACED_API(DataType, GetDataType)

  DEF_TRACED_API(const Quantization&, GetQuantization)

  // DEF_TRACED_API(TensorSpec&, GetSpec)
  TensorSpec& GetSpec() {
    return spec_;
  }


  // DEF_TRACED_API(uint32_t, GetId)
  DEF_SIMPLE_UNTRACED_API(uint32_t, GetId)

  #define SPECIAL_MACRO_(params)                                               \
    uint32_t count = BOOST_PP_SEQ_ELEM(1, params);                             \
    uint32_t offset =                                                          \
        Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params),                        \
                              sizeof(char), count);                            \
    LOGGING_POINTER_MSG(offset, count, 0)

  DEF_MEMFN(bool, CopyDataToTensor, ((const void*))((uint32_t)(0)),
            SPECIAL_MACRO_
  )

  #undef SPECIAL_MACRO_

  DEF_TRACED_API(bool, CopyDataFromTensor, [&]() {
    uint32_t count = impl_->GetSpec().GetByteSize();
    uint32_t offset = Tracer::dump_data(boost::hana::at_c<0>(params_tuple),
        sizeof(char), count);
    LOGGING_POINTER_MSG(offset, count, 0)
  })

  DEF_TRACED_API(bool, FlushCacheForHandle)

  DEF_TRACED_API(bool, InvalidateCacheForHandle)

  DEF_MEMFN(void*, map, ((bool)(false))
  )

  DEF_TRACED_API(void, unmap)

  DEF_TRACED_API(bool, IsPlaceHolder)

  DEF_TRACED_API(bool, IsConstTensor)

  DEF_TRACED_API(bool, SaveTensorToTextByFp32)

  DEF_TRACED_API(void*, ConvertTensorToData)

 private:
  TensorSpec spec_;
};

} /* namespace trace */


#endif // TIM_EXPERIMENTAL_TRACE_TVX_TENSOR_H_
