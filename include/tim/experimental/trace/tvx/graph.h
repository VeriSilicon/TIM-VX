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
#ifndef TIM_EXPERIMENTAL_TRACE_TVX_GRAPH_H_
#define TIM_EXPERIMENTAL_TRACE_TVX_GRAPH_H_
#include "tim/vx/graph.h"
#include "tim/experimental/trace/tvx/ops.h"
#include "tim/experimental/trace/tvx/tensor.h"
#include "tim/experimental/trace/tracer.h"

namespace trace {

namespace target = ::tim::vx;

#define DECL_CREATE_OP_IMPL_(r, _, opname)                                     \
  template <class... Params>                                                   \
  std::shared_ptr<trace::ops::opname> CreateOperationImpl(                     \
      BOOST_PP_CAT(ops::_VSI_Tag_of_, opname), Params... params);

#define DECL_CREATE_OPS(ops_seq)                                               \
  BOOST_PP_SEQ_FOR_EACH(DECL_CREATE_OP_IMPL_, _, ops_seq)

#define TO_STRING_(expr) #expr

#define DEF_CREATE_OP_IMPL_(r, _, opname)                                      \
template <class... Params>                                                     \
std::shared_ptr<trace::ops::opname> Graph::CreateOperationImpl(                \
      BOOST_PP_CAT(ops::_VSI_Tag_of_, opname), Params... params) {             \
  std::string this_obj_name = TraceGetObjName();                               \
  std::string obj_name =                                                       \
      Tracer::allocate_obj_name(std::string(TO_STRING_(opname)) + "_");        \
  Tracer::logging_msg(                                                         \
      "auto %s = %s->CreateOperation<%s::ops::%s>(", obj_name.c_str(),         \
      this_obj_name.c_str(), target_namespace_name_, TO_STRING_(opname));      \
  Tracer::clear_params_log_cache();                                            \
  boost::hana::tuple<Params...> params_tuple = {params...};                    \
  boost::hana::for_each(params_tuple, [&] (auto x) {                           \
    Tracer::logging_param(x, -1);                                              \
  });                                                                          \
  Tracer::dump_params_log_cache();                                             \
  Tracer::logging_msg(");\n");                                                 \
  auto CreateOpImpl = [&] (auto&&... ts) {                                     \
      return impl_->CreateOperation<target::ops::opname>(ts...);               \
  };                                                                           \
  auto proc_param_lambda = [] (auto&& t) {                                     \
    return Tracer::proc_param(t);                                              \
  };                                                                           \
  auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);  \
  auto op_impl_sp = boost::hana::unpack(params_impl, CreateOpImpl);            \
  auto op_sp = std::make_shared<trace::ops::opname>(op_impl_sp);               \
  Tracer::insert_obj_name(static_cast<void*>(op_impl_sp.get()), obj_name);     \
  return op_sp;                                                                \
}

#define DEF_CREATE_OP_IMPLS(ops_seq)                                           \
    BOOST_PP_SEQ_FOR_EACH(DEF_CREATE_OP_IMPL_, _, ops_seq)

#define SPECIALIZATION_CREATE_OP(opname, args_desc, SPECIAL_MACRO_)            \
template <>                                                                    \
inline std::shared_ptr<trace::ops::opname> Graph::CreateOperation(             \
    ARGS_DESC_TO_DECLARATION(args_desc)) {                                     \
  std::string this_obj_name = TraceGetObjName();                               \
  std::string obj_name = Tracer::allocate_obj_name(std::string(#opname) + "_");\
  Tracer::push_back_msg_cache(                                                 \
      "auto " + obj_name + " = " + this_obj_name +                             \
      "->CreateOperation<target::ops::" + #opname + ">(");                     \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
  auto op_impl_sp = impl_->CreateOperation<target::ops::opname>(               \
          SEQ_TO_VARIADICS(ARGS_DESC_TO_PARAMS(args_desc)));                   \
  auto op_sp = std::make_shared<trace::ops::opname>(op_impl_sp);               \
  Tracer::insert_obj_name(static_cast<void*>(op_impl_sp.get()), obj_name);     \
  return op_sp;                                                                \
}

struct Graph : public TraceClassBase<target::Graph> {
  DEF_INTERFACE_CONSTRUCTOR(Graph)

  #define SPECIAL_MACRO_(params)                                               \
    if (BOOST_PP_SEQ_ELEM(1, params) == nullptr) {                             \
      Tracer::insert_params_log_cache("nullptr", 1);                           \
    } else {                                                                   \
      uint32_t count =                                                         \
          BOOST_PP_SEQ_ELEM(0, params).TraceGetImpl().GetByteSize();           \
      uint32_t offset =                                                        \
          Tracer::dump_data(                                                   \
              BOOST_PP_SEQ_ELEM(1, params), sizeof(char), count);              \
      LOGGING_POINTER_MSG(offset, count, 1)                                    \
    }

  // DEF_MEMFN_SP(Tensor,
  //              CreateTensor,
  //              ((const TensorSpec&))((const void*)(nullptr)),
  //              SPECIAL_MACRO_)
  std::shared_ptr<Tensor> CreateTensor(
      const TensorSpec &param_0, const void *param_1 = nullptr) {
    std::string this_obj_name = TraceGetObjName();
    std::string obj_name =
        Tracer::allocate_obj_name(Tracer::get_obj_prefix("Tensor"));
    Tracer::push_back_msg_cache("auto " + obj_name + " = " + this_obj_name
        + "->" + __FUNCTION__ + "(");
    Tracer::init_params_log_cache(2);
    Tracer::logging_param(param_0, 0);
    Tracer::logging_param(param_1, 1);
    SPECIAL_MACRO_((param_0)(param_1))
    Tracer::pop_params_log_cache();
    Tracer::amend_last_msg_cache(");\n");
    Tracer::msg_cache_sync_to_file();

    // ToDo: the feature to use fake network weights need further refine.
    #if 1 /* if use fake input data */
    if (param_0.TraceGetImpl().GetTensorAttribute() == 
        TensorAttribute::CONSTANT && param_1 == nullptr) {
      auto fake_vec_name = Tracer::allocate_obj_name("fake_vec_");
      switch (param_0.TraceGetImpl().GetDataType()) {
      case DataType::INT32:
        Tracer::logging_msg("std::vector<int> %s(%d, 42);\n"
            "%s->CopyDataToTensor(%s.data());\n",
            fake_vec_name.c_str(), (int)param_0.TraceGetImpl().GetElementNum(),
            obj_name.c_str(), fake_vec_name.c_str());
        break;
      case DataType::FLOAT32:
        Tracer::logging_msg("std::vector<float> %s(%d, 0.42);\n"
            "%s->CopyDataToTensor(%s.data());\n",
            fake_vec_name.c_str(), (int)param_0.TraceGetImpl().GetElementNum(),
            obj_name.c_str(), fake_vec_name.c_str());
        break;
      case DataType::FLOAT16:
        Tracer::logging_msg("std::vector<int16_t> %s(%d, 0);\n"
            "%s->CopyDataToTensor(%s.data());\n",
            fake_vec_name.c_str(), (int)param_0.TraceGetImpl().GetElementNum(),
            obj_name.c_str(), fake_vec_name.c_str());
        break;
      case DataType::BOOL8:
        Tracer::logging_msg("std::vector<char> %s(%d, 0);\n"
            "%s->CopyDataToTensor(%s.data());\n",
            fake_vec_name.c_str(), (int)param_0.TraceGetImpl().GetElementNum(),
            obj_name.c_str(), fake_vec_name.c_str());
        break;
      default:
        TCLOGE("Unimplemented fake data type:%d\n",
            (int)param_0.TraceGetImpl().GetDataType());
        break;
      }
    }
    #endif /* if use fake input data */
  
    auto obj_impl_sp = impl_->CreateTensor(
        Tracer::proc_param(param_0), Tracer::proc_param(param_1));
    auto obj_sp = std::make_shared<Tensor>(obj_impl_sp, param_0);
    Tracer::insert_obj_name(static_cast<void *>(obj_impl_sp.get()), obj_name);
    tensor_sp_keeper_[obj_impl_sp] = obj_sp; // need keep obj_sp alive.
    spec_keeper_.push_back(param_0);
    return obj_sp;
  }

  // DEF_MEMFN_SP(Tensor,
  //              CreateIOTensor,
  //              ((const TensorSpec&))((void*)(nullptr)),
  //              SPECIAL_MACRO_)
  std::shared_ptr<Tensor> CreateIOTensor(
      const TensorSpec &param_0, void *param_1 = nullptr) {
    std::string this_obj_name = TraceGetObjName();
    std::string obj_name =
        Tracer::allocate_obj_name(Tracer::get_obj_prefix("Tensor"));
    Tracer::push_back_msg_cache("auto " + obj_name + " = " + this_obj_name
        + "->" + __FUNCTION__ + "(");
    Tracer::init_params_log_cache(2);
    Tracer::logging_param(param_0, 0);
    Tracer::logging_param(param_1, 1);
    SPECIAL_MACRO_((param_0)(param_1))
    Tracer::pop_params_log_cache();
    Tracer::amend_last_msg_cache(");\n");
    Tracer::msg_cache_sync_to_file();
    auto obj_impl_sp = impl_->CreateIOTensor(
        Tracer::proc_param(param_0), Tracer::proc_param(param_1));
    auto obj_sp = std::make_shared<Tensor>(obj_impl_sp);
    Tracer::insert_obj_name(static_cast<void *>(obj_impl_sp.get()), obj_name);
    tensor_sp_keeper_[obj_impl_sp] = obj_sp; // need keep obj_sp alive.
    spec_keeper_.push_back(param_0);
    return obj_sp;
  }

  #undef SPECIAL_MACRO_

  // DEF_MEMFN_SP(Tensor, CreateTensor,
  //              ((const TensorSpec&))((const DmaBufferDesc&))
  // )
  std::shared_ptr<Tensor> CreateTensor(
      const TensorSpec &param_0, const DmaBufferDesc &param_1) {
    std::string this_obj_name = TraceGetObjName();
    std::string obj_name =
        Tracer::allocate_obj_name(Tracer::get_obj_prefix("Tensor"));
    Tracer::logging_msg("auto %s = %s->%s(", obj_name.c_str(),
        this_obj_name.c_str(), __FUNCTION__);
    Tracer::init_params_log_cache(2);
    Tracer::logging_param(param_0, 0);
    Tracer::logging_param(param_1, 1);
    Tracer::dump_params_log_cache();
    Tracer::logging_msg(");\n");
    auto obj_impl_sp = impl_->CreateTensor(Tracer::proc_param(param_0),
        Tracer::proc_param(param_1));
    auto obj_sp = std::make_shared<Tensor>(obj_impl_sp);
    Tracer::insert_obj_name(static_cast<void *>(obj_impl_sp.get()), obj_name);
    tensor_sp_keeper_[obj_impl_sp] = obj_sp; // need keep obj_sp alive.
    spec_keeper_.push_back(param_0);
    return obj_sp;
  }

  // DEF_MEMFN_SP(Tensor, CreateTensorPlaceHolder)
  std::shared_ptr<Tensor> CreateTensorPlaceHolder() {
    std::string this_obj_name = TraceGetObjName();
    std::string obj_name =
        Tracer::allocate_obj_name(Tracer::get_obj_prefix("Tensor"));
    Tracer::logging_msg("auto %s = %s->%s();\n", obj_name.c_str(),
        this_obj_name.c_str(), __FUNCTION__);
    auto obj_impl_sp = impl_->CreateTensorPlaceHolder();
    auto obj_sp = std::make_shared<Tensor>(obj_impl_sp);
    Tracer::insert_obj_name(static_cast<void *>(obj_impl_sp.get()), obj_name);
    tensor_sp_keeper_[obj_impl_sp] = obj_sp; // need keep obj_sp alive.
    return obj_sp;
  }

  DEF_TRACED_API(bool, Compile)

  // DEF_TRACED_API(bool, CompileToBinary, [&] () {
  //   if (boost::hana::at_c<0>(params_tuple) == nullptr) {
  //     auto nbg_size_var = Tracer::allocate_obj_name("nbg_size_");
  //     Tracer::insert_obj_name(boost::hana::at_c<1>(params_tuple), nbg_size_var);
  //     Tracer::insert_before_last_msg_cache(
  //       "size_t " + nbg_size_var + " = -1;\n");
  //     Tracer::insert_params_log_cache("nullptr", 0);
  //   } else {
  //     uint32_t count = *boost::hana::at_c<1>(params_tuple);
  //     uint32_t offset = Tracer::dump_data(boost::hana::at_c<0>(params_tuple),
  //         sizeof(char), count);
  //     LOGGING_POINTER_MSG(offset, count, 0)
  //   }
  //   Tracer::insert_params_log_cache(
  //       "&" + Tracer::get_obj_name(boost::hana::at_c<1>(params_tuple)), 1);
  // })

  #define SPECIAL_MACRO_(params)                                               \
    if (BOOST_PP_SEQ_ELEM(0, params) == nullptr) {                             \
      std::string size_name = Tracer::allocate_obj_name("nbg_size_");          \
      Tracer::insert_obj_name(BOOST_PP_SEQ_ELEM(1, params), size_name);        \
      Tracer::insert_before_last_msg_cache(                                    \
          "size_t " + size_name + " = -1;\n");                                 \
      Tracer::insert_params_log_cache("nullptr", 0);                           \
    } else {                                                                   \
      uint32_t data_length = *BOOST_PP_SEQ_ELEM(1, params);                    \
      uint32_t offset = Tracer::dump_data(                                     \
          BOOST_PP_SEQ_ELEM(0, params), sizeof(char), data_length);            \
      LOGGING_POINTER_MSG(offset, data_length, 0)                              \
    }                                                                          \
    Tracer::insert_params_log_cache(                                           \
        "&" + Tracer::get_obj_name(BOOST_PP_SEQ_ELEM(1, params)), 1);

  DEF_MEMFN(bool,
            CompileToBinary,
            ((void*))((size_t*)),
            SPECIAL_MACRO_)
  
  #undef SPECIAL_MACRO_

  DEF_TRACED_API(bool, Run)

  template <class OpType, class... Params>
  std::shared_ptr<OpType> CreateOperation(Params... params) {
    return CreateOperationImpl(
      typename ops::TagDispatchTrait<OpType>::tag {}, params...);
  }

  const std::vector<std::shared_ptr<Tensor>> InputsTensor() {
    std::vector<std::shared_ptr<Tensor>> ret;
    for (auto& x : impl_->InputsTensor()) {
      ret.push_back(tensor_sp_keeper_[x]);
    }
    return ret;
  }

  const std::vector<std::shared_ptr<Tensor>> OutputsTensor() {
    std::vector<std::shared_ptr<Tensor>> ret;
    for (auto& x : impl_->OutputsTensor()) {
      ret.push_back(tensor_sp_keeper_[x]);
    }
    return ret;
  }

  DEF_TRACED_API(void, UpdateTensorConsumersMap)

  DEF_TRACED_API(void, UpdateTensorProducerMap)

  // DEF_TRACED_API(const std::vector<std::shared_ptr<Operation>>, GetConsumersOp)

  // DEF_TRACED_API(std::shared_ptr<Operation>, GetProducerOp)

  DEF_TRACED_API(void, PrintGraph)

  // DEF_TRACED_API(const std::vector<std::shared_ptr<Tensor>>, GetConstantInputs)

  // DEF_TRACED_API(const std::vector<std::shared_ptr<Operation>>, GetOpVector)

 private:
  std::unordered_map<std::shared_ptr<target::Tensor>, std::shared_ptr<Tensor>>
      tensor_sp_keeper_;
  std::vector<TensorSpec> spec_keeper_;

  DECL_CREATE_OPS(TVX_OPS_SEQ)
  DECL_CREATE_OP_IMPL_(_, _, Pad)   // there are enums defined in the op
  DECL_CREATE_OP_IMPL_(_, _, PadV2) // there are enums defined in the op

};
// For the NBG op, has no good enough way to get the nbg buffer size in the
// scope of `CreateOperation<tim::vx::ops::NBG>(buf, inp_num, out_num)`, so
// we enable `export VIV_VX_ENABLE_DUMP_NBG=1` to dump nbg at first, then read
// it's size.
// The best solution is make nbg buf self-analytic the size.
// ToDo: push nbg team provide an api to self-analytic the nbg size.
#define SPECIAL_MACRO_(params)                                                 \
  std::string buf_name = Tracer::allocate_obj_name("nbg_buf_vec_");            \
  FILE* nbg_dumped = fopen("network_binary_graph.nb", "r");                    \
  fseek(nbg_dumped, 0L, SEEK_END);                                             \
  uint32_t count = ftell(nbg_dumped);                                          \
  fclose(nbg_dumped);                                                          \
  uint32_t offset = Tracer::dump_data(                                         \
      BOOST_PP_SEQ_ELEM(0, params), sizeof(char), count);                      \
  Tracer::insert_before_last_msg_cache("std::vector<char> " + buf_name +       \
      " = trace::Replayer::get_vector<char>(" + std::to_string(offset) +       \
      "," + std::to_string(count) + ");\n");                                   \
  Tracer::insert_params_log_cache(buf_name + ".data()", 0);

SPECIALIZATION_CREATE_OP(NBG, ((const char*))((size_t))((size_t)),
                          SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

DEF_CREATE_OP_IMPLS(TVX_OPS_SEQ)
DEF_CREATE_OP_IMPL_(_, _, Pad)
DEF_CREATE_OP_IMPL_(_, _, PadV2)

} /* namespace trace */

#endif // TIM_EXPERIMENTAL_TRACE_TVX_GRAPH_H_
