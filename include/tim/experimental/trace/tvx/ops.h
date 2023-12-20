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
#ifndef TIM_EXPERIMENTAL_TRACE_TVX_OPS_H_
#define TIM_EXPERIMENTAL_TRACE_TVX_OPS_H_
#include "tim/vx/operation.h"
#include "tim/vx/ops.h"
#include "tim/experimental/trace/tvx/tensor.h"
#include "tim/experimental/trace/tracer.h"

#define TVX_OPS_SEQ                                                            \
  (Relu)                                                                       \
  (Relu1)                                                                      \
  (Relu6)                                                                      \
  (Tanh)                                                                       \
  (Sigmoid)                                                                    \
  (Swish)                                                                      \
  (HardSwish)                                                                  \
  (Mish)                                                                       \
  (SoftRelu)                                                                   \
  (Sign)                                                                       \
  (SoftSign)                                                                   \
  (Elu)                                                                        \
  (Prelu)                                                                      \
  (HardSigmoid)                                                                \
  (LeakyRelu)                                                                  \
  (Linear)                                                                     \
  (Gelu)                                                                       \
  (Selu)                                                                       \
  (Celu)                                                                       \
  (AddN)                                                                       \
  (ArgMin)                                                                     \
  (ArgMax)                                                                     \
  (Batch2Space)                                                                \
  (BatchNorm)                                                                  \
  (BidirectionalSequenceRnn)                                                   \
  (BidirectionalSequenceRnnExt)                                                \
  (Broadcast)                                                                  \
  (Clip)                                                                       \
  (Concat)                                                                     \
  (Conv1d)                                                                     \
  (Conv2d)                                                                     \
  (DeConv1d)                                                                   \
  (DeConv2d)                                                                   \
  (DepthToSpace)                                                               \
  (Dropout)                                                                    \
  (Minimum)                                                                    \
  (Maximum)                                                                    \
  (Add)                                                                        \
  (Sub)                                                                        \
  (Pow)                                                                        \
  (FloorDiv)                                                                   \
  (Multiply)                                                                   \
  (Div)                                                                        \
  (Erf)                                                                        \
  (FullyConnected)                                                             \
  (Gather)                                                                     \
  (GatherElements)                                                             \
  (GatherNd)                                                                   \
  (GroupedConv2d)                                                              \
  (InstanceNormalization)                                                      \
  (L2Normalization)                                                            \
  (LayerNormalization)                                                         \
  (LRN)                                                                        \
  (LogicalAnd)                                                                 \
  (LogicalOr)                                                                  \
  (LogSoftmax)                                                                 \
  (Matmul)                                                                     \
  (MaxpoolWithArgmax)                                                          \
  (MaxpoolWithArgmax2)                                                         \
  (MaxpoolGrad)                                                                \
  (MaxUnpool2d)                                                                \
  (Moments)                                                                    \
  (OneHot)                                                                     \
  /* (Pad) enum defined in class */                                            \
  /* (PadV2) enum defined in class */                                          \
  (Pool1d)                                                                     \
  (Pool2d)                                                                     \
  (ReduceMin)                                                                  \
  (ReduceMax)                                                                  \
  (ReduceAny)                                                                  \
  (ReduceAll)                                                                  \
  (ReduceProd)                                                                 \
  (ReduceMean)                                                                 \
  (ReduceSum)                                                                  \
  (Greater)                                                                    \
  (GreaterOrEqual)                                                             \
  (Less)                                                                       \
  (LessOrEqual)                                                                \
  (NotEqual)                                                                   \
  (Equal)                                                                      \
  (Reorg)                                                                      \
  (Reshape)                                                                    \
  (Resize1d)                                                                   \
  (Resize)                                                                     \
  (Reverse)                                                                    \
  (RNNCell)                                                                    \
  (RoiAlign)                                                                   \
  (RoiPool)                                                                    \
  (ScatterND)                                                                  \
  (Select)                                                                     \
  (ShuffleChannel)                                                             \
  (DataConvert)                                                                \
  (Neg)                                                                        \
  (Abs)                                                                        \
  (Sin)                                                                        \
  (Exp)                                                                        \
  (Log)                                                                        \
  (Sqrt)                                                                       \
  (Rsqrt)                                                                      \
  (Square)                                                                     \
  (LogicalNot)                                                                 \
  (Floor)                                                                      \
  (Ceil)                                                                       \
  (Round)                                                                      \
  (Cast)                                                                       \
  (Rcp)                                                                        \
  (SignalFrame)                                                                \
  (Slice)                                                                      \
  (Softmax)                                                                    \
  (Space2Batch)                                                                \
  (SpaceToDepth)                                                               \
  (SpatialTransformer)                                                         \
  (Split)                                                                      \
  (Squeeze)                                                                    \
  (Stack)                                                                      \
  (StridedSlice)                                                               \
  (Svdf)                                                                       \
  (Tile)                                                                       \
  (Transpose)                                                                  \
  (UnidirectionalSequenceLstm)                                                 \
  (UnidirectionalSequenceRnn)                                                  \
  (UnidirectionalSequenceRnnExt)                                               \
  (Unstack)                                                                    \
  (Conv3d)                                                                     \
  (CumSum)                                                                     \
  (LocalResponseNormalization)                                                 \
  (CustomOpBase)                                                               \
  (Topk)                                                                       \
  (BidirectionalSequenceLstm)                                                  \
  (HashtableLookup)                                                            \
  (EmbeddingLookup)                                                            \
  (NBG)

namespace trace {

namespace target = ::tim::vx;

struct Operation : public TraceClassBase<target::Operation> {
  DEF_INTERFACE_CONSTRUCTOR(Operation)

  DEF_TRACED_API(std::shared_ptr<Operation>, Clone)

  // unfixed issue when use DEF_TRACED_API to define BindOutput
  DEF_INPLACE_MEMFN(Operation&, BindInput, ((const std::shared_ptr<Tensor>&)))
  DEF_INPLACE_MEMFN(Operation&, BindOutput, ((const std::shared_ptr<Tensor>&)))
  // DEF_TRACED_API(Operation&, BindInput)
  // template <class R = Operation &, class... Args>
  // typename std::enable_if_t<is_traced_obj<R>::value, R> BindInput(Args... params)
  // {
  //   std::string this_obj_name = TraceGetObjName();
  //   Tracer::push_back_msg_cache(this_obj_name + "->" + __FUNCTION__ + "(");
  //   Tracer::clear_params_log_cache();
  //   boost::hana::tuple<Args...> params_tuple = {params...};
  //   boost::hana::for_each(params_tuple, [&](auto x)
  //                         { Tracer::logging_param(x, -1); });
  //   Tracer::pop_params_log_cache();
  //   Tracer::amend_last_msg_cache(");\n");
  //   Tracer::msg_cache_sync_to_file();
  //   auto api_impl = [&](auto &&...ts)
  //   { return &(impl_->BindInput(ts...)); };
  //   auto proc_param_lambda = [](auto &&t)
  //   { return Tracer::proc_param(t); };
  //   auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);
  //   auto ret_impl_p = boost::hana::unpack(params_impl, api_impl);
  //   return *this;
  //   // return *static_cast<std::decay_t<R> *>(Tracer::get_traced_obj(ret_impl_p));
  // }
  // DEF_TRACED_API(Operation&, BindOutput)
  // template <class R = Operation &, class... Args>
  // typename std::enable_if_t<is_traced_obj<R>::value, R> BindOutput(Args... params)
  // {
  //   std::string this_obj_name = TraceGetObjName();
  //   Tracer::push_back_msg_cache(this_obj_name + "->" + __FUNCTION__ + "(");
  //   Tracer::clear_params_log_cache();
  //   boost::hana::tuple<Args...> params_tuple = {params...};
  //   boost::hana::for_each(params_tuple, [&](auto x)
  //                         { Tracer::logging_param(x, -1); });
  //   Tracer::pop_params_log_cache();
  //   Tracer::amend_last_msg_cache(");\n");
  //   Tracer::msg_cache_sync_to_file();
  //   auto api_impl = [&](auto &&...ts)
  //   { return &(impl_->BindOutput(ts...)); };
  //   auto proc_param_lambda = [](auto &&t)
  //   { return Tracer::proc_param(t); };
  //   auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);
  //   auto ret_impl_p = boost::hana::unpack(params_impl, api_impl);
  //   return *this;
  //   // return *static_cast<std::decay_t<R> *>(Tracer::get_traced_obj(ret_impl_p));
  // }

  // compiler can not deduce the <brace-enclosed initializer list> type for
  // template variadic, so can't use DEF_TRACED_API to define BindInputs/Outputs
  // DEF_INPLACE_MEMFN(Operation&, BindInputs,
  //                   ((const std::vector<std::shared_ptr<Tensor>>&))
  // )
  Operation &BindInputs(const std::vector<std::shared_ptr<Tensor>> &param_0)
  {
    std::string this_obj_name = TraceGetObjName();
    Tracer::logging_msg("%s->%s(", this_obj_name.c_str(), __FUNCTION__);
    Tracer::init_params_log_cache(1);
    Tracer::logging_param(param_0, 0);
    Tracer::dump_params_log_cache();
    Tracer::logging_msg(");\n");
    impl_->BindInputs(Tracer::proc_param(param_0));
    return *this;
  }
  // DEF_INPLACE_MEMFN(Operation&, BindOutputs,
  //                   ((const std::vector<std::shared_ptr<Tensor>>&))
  // )
  Operation &BindOutputs(const std::vector<std::shared_ptr<Tensor>> &param_0)
  {
    std::string this_obj_name = TraceGetObjName();
    Tracer::logging_msg("%s->%s(", this_obj_name.c_str(), __FUNCTION__);
    Tracer::init_params_log_cache(1);
    Tracer::logging_param(param_0, 0);
    Tracer::dump_params_log_cache();
    Tracer::logging_msg(");\n");
    impl_->BindOutputs(Tracer::proc_param(param_0));
    return *this;
  }
  DEF_INPLACE_MEMFN(Operation&, SetRoundingPolicy, 
                    ((OverflowPolicy)(OverflowPolicy::SATURATE))
                    ((RoundingPolicy)(RoundingPolicy::RTNE))
                    ((RoundType)(RoundType::FLOOR))
                    ((uint32_t)(0))
  )

  // do not support unique_ptr and OpImpl by now
  // DEF_TRACED_API(std::unique_ptr<OpImpl>&, impl)

  // do not support unique_ptr and OpImpl by now
  // DEF_TRACED_API(const std::unique_ptr<OpImpl>&, impl)

  DEF_TRACED_API(const std::vector<std::shared_ptr<Tensor>>, ConstantInputsTensor)

 protected:
  DEF_TRACED_API(bool, IsAllInputsConst)
};

} /* namespace trace */

namespace trace {
namespace ops {

struct DefaultTag {};

template<class T>
struct TagDispatchTrait {
  using tag = DefaultTag;
};

#define DEF_TIMVX_OP_AND_TAG_IMPL_(r, _, op)                                   \
struct op : Operation {                                                        \
  op(const std::shared_ptr<target::ops::op>& impl) : Operation(impl) {}        \
};                                                                             \
struct BOOST_PP_CAT(_VSI_Tag_of_, op) {};                                      \
template<>                                                                     \
struct TagDispatchTrait<op> {                                                  \
  using tag = BOOST_PP_CAT(_VSI_Tag_of_, op);                                  \
};

#define DEF_TIMVX_OPS_AND_TAGS(ops)                                            \
  BOOST_PP_SEQ_FOR_EACH(DEF_TIMVX_OP_AND_TAG_IMPL_, _, ops)

DEF_TIMVX_OPS_AND_TAGS(TVX_OPS_SEQ)

// DEF_TIMVX_OP_AND_TAG_IMPL_(_, _, Pad)
struct Pad : Operation {
  // must be used as tvx::ops::Pad::pad_mode_type::PAD_MODE_CONSTANT, but not
  // tvx::ops::Pad::PAD_MODE_CONSTANT
  using pad_mode_type = target::ops::Pad::pad_mode_type;
  Pad(const std::shared_ptr<target::ops::Pad> &impl) : Operation(impl) {}
};
struct _VSI_Tag_of_Pad {};
template <>
struct TagDispatchTrait<Pad> {
  using tag = _VSI_Tag_of_Pad;
};

// DEF_TIMVX_OP_AND_TAG_IMPL_(_, _, PadV2)
struct PadV2 : Operation {
  using pad_mode_type = target::ops::PadV2::pad_mode_type;
  PadV2(const std::shared_ptr<target::ops::PadV2> &impl) : Operation(impl) {}
};
struct _VSI_Tag_of_PadV2 {};
template <>
struct TagDispatchTrait<PadV2> {
  using tag = _VSI_Tag_of_PadV2;
};

} /* namespace ops */
} /* namespace trace */

#endif // TIM_EXPERIMENTAL_TRACE_TVX_OPS_H_
