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
#ifndef TIM_EXPERIMENTAL_TRACE_TRACER_H_
#define TIM_EXPERIMENTAL_TRACE_TRACER_H_

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <iostream>
#include <list>
#include <mutex>
#include <vector>
#include <array>
#include <memory>
#include <sstream>
#include <algorithm>
#include <type_traits>
#include <unordered_map>

#include <boost/type_index.hpp>
#include <boost/preprocessor/seq/size.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/subseq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/variadic/size.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/unpack.hpp>

#include "tim/experimental/trace/replayer.h"

#define TRACE_LOG_NAME_ "trace_log.cc"
#define TRACE_BIN_FILE_ "trace_bin.bin"
#define TRACE_PREFIX_ENV_VAR_ "TRACE_DUMP_PREFIX" 

#define TCLOGE(fmt, ...) do {                                                  \
    printf("[ERROR] [%s:%s:%d]" fmt, __FILE__, __FUNCTION__, __LINE__,         \
        ##__VA_ARGS__);                                                        \
    fflush(stdout);                                                            \
  } while (0)


/********************** definitions of extra type traits **********************/
namespace trace {

template<typename...>
using void_t = void;

template <class, class = void>
struct is_fundamental_vector : std::false_type {};

template <class T>
struct is_fundamental_vector<std::vector<T>> {
  static constexpr bool value = std::is_fundamental<T>::value;
};

template <class, class = void>
struct is_fundamental_array : std::false_type {};

template <class T, std::size_t N>
struct is_fundamental_array<std::array<T, N>> {
  static constexpr bool value = std::is_fundamental<T>::value;
};

template <class T>
struct is_fundamental_pointer : std::integral_constant<bool,
    std::is_pointer<T>::value &&
    std::is_fundamental<std::remove_pointer_t<T>>::value> {};

template <class, class = void>
struct is_traced_obj : std::false_type {};

template <class T>
struct is_traced_obj<T,
    void_t<decltype(std::declval<T&>().TraceGetObjName())>>
  : std::true_type {};

template <class, class = void>
struct is_traced_obj_ptr : std::false_type {};

template <class T>
struct is_traced_obj_ptr<T,
    void_t<decltype(std::declval<T&>()->TraceGetObjName())>>
  : std::true_type {};

template <class, class = void>
struct is_traced_obj_ptr_vector : std::false_type {};

template <class T>
struct is_traced_obj_ptr_vector<std::vector<T>> {
  static constexpr bool value = is_traced_obj_ptr<T>::value;
};

template <class T>
struct is_others_type : std::integral_constant<bool,
    !is_fundamental_vector<std::decay_t<T>>::value &&
    !is_fundamental_array<std::decay_t<T>>::value &&
    !std::is_enum<std::decay_t<T>>::value &&
    !std::is_fundamental<std::decay_t<T>>::value &&
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value &&
    !is_traced_obj_ptr_vector<std::decay_t<T>>::value> {};

template <class T>
struct is_not_traced_obj_like : std::integral_constant<bool,
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value &&
    !is_traced_obj_ptr_vector<std::decay_t<T>>::value> {};

} /* namespace trace */


/**************************** definition of tracer ****************************/
namespace trace {

class Tracer {
  static std::unordered_map<const void*, std::string> obj_names_;
  static std::unordered_map<std::string, std::string> objs_prefix_;
  static std::vector<std::string> params_log_cache_;
  static std::list<std::string> msg_cache_;
  static std::unordered_map<const void*, void*> target2trace_map_;
  static FILE* file_trace_log_;
  static FILE* file_trace_bin_;
  static std::mutex log_mtx_;
  static std::mutex bin_mtx_;

  static FILE* open_file(const char* file_name);

 public:
  static void logging_msg(const char* format, ...);

  static uint32_t dump_data(const void* data, size_t byte_size, size_t count);

  static std::string allocate_obj_name(const std::string& prefix = "obj_");

  static inline void insert_obj_name(
      const void* obj_ptr, const std::string& obj_name) {
    obj_names_[obj_ptr] = obj_name;
  }

  static inline void insert_traced_obj(const void* p_target, void* p_traced) {
    target2trace_map_.insert({p_target, p_traced});
  }

  static inline void* get_traced_obj(const void* p_target) {
    return target2trace_map_[p_target];
  }

  static inline std::string& get_obj_name(const void* obj) {
    return obj_names_[obj];
  }

  static inline std::string& get_obj_prefix(const std::string cls_name) {
    return objs_prefix_[cls_name];
  }

  static inline void push_back_msg_cache(const std::string& msg) {
    msg_cache_.push_back(msg);
  }

  static inline void amend_last_msg_cache(const std::string& msg) {
    if (msg_cache_.empty()) {
      TCLOGE("Can't amend sub_msg, because msg cache is empty!\n");
    }
    msg_cache_.back() += msg;
  }

  static inline void insert_before_last_msg_cache(const std::string& msg) {
    msg_cache_.insert(--msg_cache_.end(), msg);
  }

  static inline void msg_cache_sync_to_file() {
    while (!msg_cache_.empty()) {
      logging_msg(msg_cache_.front().c_str());
      msg_cache_.pop_front();
    }
  }

  static inline void clear_params_log_cache() {
    params_log_cache_.clear();
  }

  static inline void init_params_log_cache(uint32_t params_size) {
    params_log_cache_.clear();
    params_log_cache_.resize(params_size);
  }

  static inline void append_params_log_cache(std::string param_log) {
    params_log_cache_.push_back(param_log);
  }

  static inline void insert_params_log_cache(
      std::string param_log, uint32_t idx) {
    if (idx != static_cast<uint32_t>(-1)) {
      params_log_cache_[idx] = param_log;
    } else {
      params_log_cache_.push_back(param_log);
    }
  }

  // pop the log of params into msg cache
  static inline void pop_params_log_cache() {
    if (params_log_cache_.size() == 0)  return;
    for (uint32_t i = 0; i < params_log_cache_.size() - 1; i++) {
      amend_last_msg_cache(params_log_cache_[i] + ", ");
    }
    amend_last_msg_cache(params_log_cache_.back());
  }

  // directly dump the log of params to file
  static inline void dump_params_log_cache() {
    if (params_log_cache_.size() == 0)  return;
    for (uint32_t i = 0; i < params_log_cache_.size() - 1; i++) {
      logging_msg("%s, ", params_log_cache_[i].c_str());
    }
    logging_msg(params_log_cache_.back().c_str());
  }

  /*
   * template functions for logging parameters to log file
   */
  // default substitution
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  template <class T,
      typename std::enable_if_t<is_others_type<T>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(std::string(), idx);
  }
  #pragma GCC diagnostic pop

  // enable if T is fundamental std::vector
  template <class T,
      typename std::enable_if_t<
          is_fundamental_vector<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    uint32_t offset = dump_data(t.data(), sizeof(t[0]), t.size());
    std::string element_type =
        boost::typeindex::type_id<decltype(t[0])>().pretty_name();
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "trace::Replayer::get_vector<%s>(%u, %u)",
             element_type.c_str(), offset, (uint32_t)t.size());
    insert_params_log_cache(std::string(log_msg), idx);
  }

  // enable if T is fundamental std::array
  template <class T,
      typename std::enable_if_t<
          is_fundamental_array<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    uint32_t offset = dump_data(t.data(), sizeof(t[0]), t.size());
    std::string element_type =
        boost::typeindex::type_id<decltype(t[0])>().pretty_name();
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "trace::Replayer::get_array<%s, %d>(%u, %u)",
             element_type.c_str(), (uint32_t)t.size(), offset,
             (uint32_t)t.size());
    insert_params_log_cache(std::string(log_msg), idx);
  }

  // enable if T is enum
  template <class T,
      typename std::enable_if_t<
          std::is_enum<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    std::string enum_type =
        boost::typeindex::type_id<decltype(t)>().pretty_name();
    char log_msg[1024] = {0};
    snprintf(log_msg, 1024, "(%s)%d", enum_type.c_str(), (int)t);
    insert_params_log_cache(std::string(log_msg), idx);
  }

  // enable if T is fundamental
  template <class T,
      typename std::enable_if_t<
          std::is_fundamental<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(std::to_string(t), idx);
  }

  // enable if T is derive from TraceClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(t.TraceGetObjName(), idx);
  }

  // enable if T is pointer to object which 
  // derive from TraceClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    insert_params_log_cache(t->TraceGetObjName(), idx);
  }

  // enable if T is vector of pointer to object which 
  // derive from TraceClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr_vector<std::decay_t<T>>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {
    std::stringstream ss;
    ss << "{";
    for (uint32_t i = 0; i < t.size() - 1; i++) {
      ss << t[i]->TraceGetObjName() << ", ";
    }
    ss << t.back()->TraceGetObjName() << "}";
    insert_params_log_cache(ss.str(), idx);
  }

  /*
   * template functions for pass correct parameters to api implementation
   */
  // default substitution
  template <class T,
      typename std::enable_if_t<is_not_traced_obj_like<T>::value, int> = 0>
  static inline T&& proc_param(T&& t) {
    return std::forward<T>(t);
  }

  // enable if T is derive from TraceClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj<std::decay_t<T>>::value, int> = 0>
  static inline decltype(std::declval<T&>().TraceGetImpl())&& proc_param(
      T&& t) {
    return std::forward<T>(t).TraceGetImpl();
  }

  // enable if T is pointer to object which 
  // derive from TraceClassBase
  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr<std::decay_t<T>>::value, int> = 0>
  static inline decltype(std::declval<T&>()->TraceGetImplSp()) proc_param(
     T&& t) {
    return std::forward<T>(t)->TraceGetImplSp();
  }

  // enable if T is std::vector which element is pointer of traced object
  template <class T,
      typename std::enable_if_t<
          is_traced_obj_ptr_vector<std::decay_t<T>>::value, int> = 0>
  static inline std::vector<decltype(std::declval<T&>()[0]->TraceGetImplSp())>
      proc_param(T&& t) {
    std::vector<decltype(t[0]->TraceGetImplSp())> impl_vec;
    for (auto& x : std::forward<T>(t)) {
      impl_vec.emplace_back(x->TraceGetImplSp());
    }
    return impl_vec;
  }
};

#ifdef API_TRACER_IMPLEMENTATION
std::unordered_map<const void*, std::string> Tracer::obj_names_;
std::vector<std::string> Tracer::params_log_cache_;
std::list<std::string> Tracer::msg_cache_;
std::unordered_map<const void*, void*> Tracer::target2trace_map_;
FILE* Tracer::file_trace_log_ = Tracer::open_file(TRACE_LOG_NAME_);
FILE* Tracer::file_trace_bin_ = Tracer::open_file(TRACE_BIN_FILE_);
std::mutex Tracer::log_mtx_;
std::mutex Tracer::bin_mtx_;
std::unordered_map<std::string, std::string> Tracer::objs_prefix_ = {
    {"Quantization", "quant_"    },
    {"TensorSpec",  "spec_"      },
    {"Tensor",      "tensor_"    },
    {"Graph",       "graph_"     }
};
/* static */ FILE* Tracer::open_file(const char* file_name) {
  char* prefix = getenv(TRACE_PREFIX_ENV_VAR_);
  FILE* fp;
  char path[1024] = {0};
  if (prefix != NULL) {
    strcpy(path, prefix);
    strcat(path, file_name);
  } else {
    strcpy(path, file_name);
  }
  fp = fopen(path, "w");
  if (!fp) {
    TCLOGE("Can not open file at: %s\n", path);
  }
  return fp;
}

/* static */ std::string Tracer::allocate_obj_name(
    const std::string& prefix) {
  static std::unordered_map<std::string, uint32_t> objects_counter;
  if (objects_counter.find(prefix) == objects_counter.end()) {
    objects_counter[prefix] = 0;
  }
  return prefix + std::to_string(objects_counter[prefix]++);
}

/* static */ void Tracer::logging_msg(const char* format, ...) {
  char arg_buffer[1024] = {0};
  va_list args;
  va_start(args, format);
  vsnprintf(arg_buffer, 1024, format, args);
  va_end(args);
  // printf("%s", arg_buffer);
  if (!file_trace_log_) {
    TCLOGE("Log file do not exist\n");
  }
  std::lock_guard<std::mutex> lock(log_mtx_);
  fprintf(file_trace_log_, "%s", arg_buffer);
  fflush(file_trace_log_);
}

/* static */ uint32_t Tracer::dump_data(
    const void* data, size_t byte_size, size_t count) {
  std::lock_guard<std::mutex> lock(bin_mtx_);
  if (fwrite(data, byte_size, count, file_trace_bin_) != count) {
    TCLOGE("Write trace binary data failed!\n");
  }
  fflush(file_trace_bin_);
  static uint32_t offset = 0;
  uint32_t temp = offset;
  offset += byte_size * count;
  return temp;
}
#endif /* #ifdef API_TRACER_IMPLEMENTATION */

} /* namespace trace */

/************************ definition of TraceClassBase ************************/
namespace trace {

template <class TargetClass>
struct TraceClassBase {
  TargetClass& TraceGetImpl() const { return *impl_; }

  // temporary return rvalue to prevent compile error
  std::shared_ptr<TargetClass> TraceGetImplSp() { return impl_; }

  std::string& TraceGetObjName() const {
    return Tracer::get_obj_name(static_cast<const void*>(impl_.get()));
  }

  std::shared_ptr<TargetClass> impl_;
  // static const char* target_namespace_name_;
};

// #ifdef TARGET_NAMESPACE_NAME
// template <class TargetClass>
// const char* TraceClassBase<TargetClass>::target_namespace_name_ =
//     TARGET_NAMESPACE_NAME;
// #endif // #ifdef TARGET_NAMESPACE_NAME

} /* namespace trace */

#ifdef TARGET_NAMESPACE_NAME
static const char* target_namespace_name_ = TARGET_NAMESPACE_NAME;
#endif // #ifdef TARGET_NAMESPACE_NAME

/**************************** definitions of macros ***************************/
#define LOG_PARAM_IMPL_(r, _, i, param)                                        \
  Tracer::logging_param(param, i);

#define LOG_PARAMS(params)                                                     \
  Tracer::init_params_log_cache(BOOST_PP_SEQ_SIZE(params));                    \
  BOOST_PP_SEQ_FOR_EACH_I(LOG_PARAM_IMPL_, _, params)

#define PROC_PARAM_IMPL_COMMA_(r, _, param)                                    \
  Tracer::proc_param(param),

#define PROC_PARAM_IMPL_NO_COMMA_(param)                                       \
  Tracer::proc_param(param)

#define PROC_SINGLE_PARAM_(params)                                             \
  PROC_PARAM_IMPL_NO_COMMA_(BOOST_PP_SEQ_ELEM(0, params))

#define PROC_MULTI_PARAMS_(params)                                             \
  BOOST_PP_SEQ_FOR_EACH(                                                       \
    PROC_PARAM_IMPL_COMMA_, _,                                                 \
    BOOST_PP_SEQ_SUBSEQ(params, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params))))   \
  PROC_PARAM_IMPL_NO_COMMA_(                                                   \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(params)), params))

#define PROC_PARAMS(params)                                                    \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(params), 1),                    \
              PROC_SINGLE_PARAM_, PROC_MULTI_PARAMS_)(params)


#define NAME_A_PARAM_(r, data, i, elem) (param_##i)

#define ARGS_DESC_TO_PARAMS(descs)                                             \
  BOOST_PP_SEQ_FOR_EACH_I(NAME_A_PARAM_, _, descs)

#define IS_WITH_DEFAULT_VAL_(desc)                                             \
  BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(desc), 2)

#define SET_DEFAULT_VAL_(val) = BOOST_PP_SEQ_HEAD(val)

#define DO_NOTHING(x)

#define DECLARE_AN_ARG_COMMA_(r, names, i, desc)                               \
  BOOST_PP_SEQ_HEAD(desc) BOOST_PP_SEQ_ELEM(i, names)                          \
  BOOST_PP_IF(IS_WITH_DEFAULT_VAL_(desc), SET_DEFAULT_VAL_, DO_NOTHING)        \
    (BOOST_PP_SEQ_TAIL(desc)),

#define DECLARE_AN_ARG_NO_COMMA_(name, desc)                                   \
  BOOST_PP_SEQ_HEAD(desc) name                                                 \
  BOOST_PP_IF(IS_WITH_DEFAULT_VAL_(desc), SET_DEFAULT_VAL_, DO_NOTHING)        \
    (BOOST_PP_SEQ_TAIL(desc))

#define SINGLE_ARG_DESC_TO_DECLARATION_(desc)                                  \
  DECLARE_AN_ARG_NO_COMMA_(                                                    \
    BOOST_PP_SEQ_ELEM(0, ARGS_DESC_TO_PARAMS(desc)),                           \
    BOOST_PP_SEQ_ELEM(0, desc))

#define MULTI_ARGS_DESC_TO_DECLARATION_(descs)                                 \
  BOOST_PP_SEQ_FOR_EACH_I(DECLARE_AN_ARG_COMMA_,                               \
    BOOST_PP_SEQ_SUBSEQ(ARGS_DESC_TO_PARAMS(descs),                            \
                        0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs))),            \
    BOOST_PP_SEQ_SUBSEQ(descs, 0,                                              \
                        BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs))))               \
  DECLARE_AN_ARG_NO_COMMA_(                                                    \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs)),                  \
                      ARGS_DESC_TO_PARAMS(descs)),                             \
    BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(descs)), descs))

#define ARGS_DESC_TO_DECLARATION(descs)                                        \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_SEQ_SIZE(descs), 1),                     \
              SINGLE_ARG_DESC_TO_DECLARATION_,                                 \
              MULTI_ARGS_DESC_TO_DECLARATION_)(descs)

#define TO_VARIADIC_IMPL_COMMA_(r, _, elem) elem,
#define TO_VARIADIC_IMPL_NO_COMMA_(elem) elem

#define SEQ_TO_VARIADICS(seqs)                                                 \
  BOOST_PP_SEQ_FOR_EACH(TO_VARIADIC_IMPL_COMMA_, _,                            \
      BOOST_PP_SEQ_SUBSEQ(seqs, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(seqs))))     \
  TO_VARIADIC_IMPL_NO_COMMA_(                                                  \
      BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(seqs)), seqs))

#define DEF_MEMFN_SP_2_(ret_class, api_name)                                   \
  std::shared_ptr<ret_class> api_name() {                                      \
    std::string this_obj_name = TraceGetObjName();                             \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#ret_class));         \
    Tracer::logging_msg("auto %s = %s->%s();\n", obj_name.c_str(),             \
                              this_obj_name.c_str(), __FUNCTION__);            \
    auto obj_impl_sp = impl_->api_name();                                      \
    auto obj_sp = std::make_shared<ret_class>(obj_impl_sp);                    \
    Tracer::insert_obj_name(static_cast<void*>(obj_impl_sp.get()), obj_name);  \
    return obj_sp;                                                             \
  }

#define DEF_MEMFN_SP_3_(ret_class, api_name, args_desc)                        \
  std::shared_ptr<ret_class> api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {   \
    std::string this_obj_name = TraceGetObjName();                             \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#ret_class));         \
    Tracer::logging_msg("auto %s = %s->%s(", obj_name.c_str(),                 \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    Tracer::dump_params_log_cache();                                           \
    Tracer::logging_msg(");\n");                                               \
    auto obj_impl_sp = impl_->api_name(                                        \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    auto obj_sp = std::make_shared<ret_class>(obj_impl_sp);                    \
    Tracer::insert_obj_name(static_cast<void*>(obj_impl_sp.get()), obj_name);  \
    return obj_sp;                                                             \
  }

#define DEF_MEMFN_SP_4_(ret_class, api_name, args_desc, SPECIAL_MACRO_)        \
  std::shared_ptr<ret_class> api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {   \
    std::string this_obj_name = TraceGetObjName();                             \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#ret_class));         \
    Tracer::push_back_msg_cache("auto " + obj_name + " = " + this_obj_name     \
        + "->" + __FUNCTION__ + "(");                                          \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    auto obj_impl_sp = impl_->api_name(                                        \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    auto obj_sp = std::make_shared<ret_class>(obj_impl_sp);                    \
    Tracer::insert_obj_name(static_cast<void*>(obj_impl_sp.get()), obj_name);  \
    return obj_sp;                                                             \
  }

#define DEF_SIMPLE_UNTRACED_API(retval, api_name)                              \
  retval api_name() {                                                          \
    return impl_->api_name();                                                  \
  }

#define DEF_MEMFN_2_(retval, api_name)                                         \
  retval api_name() {                                                          \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::logging_msg("%s->%s();\n",                                         \
                              this_obj_name.c_str(), __FUNCTION__);            \
    return impl_->api_name();                                                  \
  }

#define DEF_MEMFN_3_(retval, api_name, args_desc)                              \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::logging_msg("%s->%s(",                                             \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    Tracer::dump_params_log_cache();                                           \
    Tracer::logging_msg(");\n");                                               \
    return impl_->api_name(                                                    \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
  }

#define DEF_MEMFN_4_(retval, api_name, args_desc, SPECIAL_MACRO_)              \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::push_back_msg_cache(                                               \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    return impl_->api_name(                                                    \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
  }

#define DEF_INPLACE_MEMFN_2_(retval, api_name)                                 \
  retval api_name() {                                                          \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::logging_msg("%s->%s();\n",                                         \
        this_obj_name.c_str(), __FUNCTION__);                                  \
    impl_->api_name();                                                         \
    return *this;                                                              \
  }

#define DEF_INPLACE_MEMFN_3_(retval, api_name, args_desc)                      \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::logging_msg("%s->%s(",                                             \
                              this_obj_name.c_str(), __FUNCTION__);            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    Tracer::dump_params_log_cache();                                           \
    Tracer::logging_msg(");\n");                                               \
    impl_->api_name(PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));              \
    return *this;                                                              \
  }

#define DEF_INPLACE_MEMFN_4_(retval, api_name, args_desc, SPECIAL_MACRO_)      \
  retval api_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                       \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::push_back_msg_cache(                                               \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    impl_->api_name(PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));              \
    return *this;                                                              \
  }

#define DEF_CONSTRUCTOR_1_(class_name)                                         \
  class_name() {                                                               \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#class_name));        \
    Tracer::logging_msg("auto %s = %s::%s();\n", obj_name.c_str(),             \
        target_namespace_name_, __FUNCTION__);                                 \
    impl_ = std::make_shared<target::class_name>();                            \
    Tracer::insert_traced_obj(                                                 \
        static_cast<void*>(impl_.get()), static_cast<void*>(this));            \
    Tracer::insert_obj_name(static_cast<void*>(impl_.get()), obj_name);        \
  }

#define DEF_CONSTRUCTOR_2_(class_name, args_desc)                              \
  class_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                            \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#class_name));        \
    Tracer::logging_msg("auto %s = %s::%s(", obj_name.c_str(),                 \
        target_namespace_name_, __FUNCTION__);                                 \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    Tracer::dump_params_log_cache();                                           \
    Tracer::logging_msg(");\n");                                               \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    Tracer::insert_traced_obj(                                                 \
        static_cast<void*>(impl_.get()), static_cast<void*>(this));            \
    Tracer::insert_obj_name(static_cast<void*>(impl_.get()), obj_name);        \
  }

#define DEF_CONSTRUCTOR_3_(class_name, args_desc, SPECIAL_MACRO_)              \
  class_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                            \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#class_name));        \
    Tracer::push_back_msg_cache(                                               \
        "auto " + obj_name + " = " + target_namespace_name_ + "::" +           \
        __FUNCTION__ + "(");                                                   \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    SPECIAL_MACRO_(ARGS_DESC_TO_PARAMS(args_desc))                             \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    Tracer::insert_traced_obj(                                                 \
        static_cast<void*>(impl_.get()), static_cast<void*>(this));            \
    Tracer::insert_obj_name(static_cast<void*>(impl_.get()), obj_name);        \
  }


/*
 * Follows code to detect empty macro variadic is from:
 * https://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments/
 */
#define __ARG16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,_14,\
  _15, ...) _15
#define __HAS_COMMA(...) __ARG16(__VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
  1, 1, 1, 0)
#define __TRIGGER_PARENTHESIS_(...) ,
#define __PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define __IS_EMPTY_CASE_0001 ,
#define __IS_EMPTY(_0, _1, _2, _3) __HAS_COMMA(                                \
  __PASTE5(__IS_EMPTY_CASE_, _0, _1, _2, _3))

#define TUPLE_IS_EMPTY(...)                                                    \
  __IS_EMPTY(                                                                  \
    /* test if there is just one argument, eventually an empty one */          \
    __HAS_COMMA(__VA_ARGS__),                                                  \
    /* test if _TRIGGER_PARENTHESIS_ together with the argument adds a comma */\
    __HAS_COMMA(__TRIGGER_PARENTHESIS_ __VA_ARGS__),                           \
    /* test if the argument together with a parenthesis adds a comma */        \
    __HAS_COMMA(__VA_ARGS__ (/*empty*/)),                                      \
    /* test if placing it between _TRIGGER_PARENTHESIS_ */                     \
    /* and the parenthesis adds a comma */                                     \
    __HAS_COMMA(__TRIGGER_PARENTHESIS_ __VA_ARGS__ (/*empty*/))                \
  )

#define EMPTY_LAMBDA_ placeholder
// BOOST_PP_VARIADIC_SIZE() variadic is empty, but it's will expand to 1.
// And there are unknow issue of tensorflow build, ##__VA_ARGS__ can't correctly 
// remove the comma before.
#define VARIADIC_SIZE_PLUS_ONE(...)                                            \
  BOOST_PP_VARIADIC_SIZE(EMPTY_LAMBDA_, ##__VA_ARGS__)
#define INVOKE_LAMBDA_(r, _, lambda) lambda();
#define INVOKE_LAMBDAS_(...)                                                   \
  BOOST_PP_IF(BOOST_PP_EQUAL(TUPLE_IS_EMPTY(__VA_ARGS__), 1),                  \
      BOOST_PP_EMPTY(), BOOST_PP_SEQ_FOR_EACH(INVOKE_LAMBDA_, _,               \
          BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

/*
 * In substitution for traced obj, api_impl got pointer retval because if
 * directly return reference will get build error of using pure virtual class
 * constructor.
 */
#define DEF_TRACED_API(Ret, api_name, ...)                                     \
  template <class R = Ret, class... Args>                                      \
  typename std::enable_if_t<is_traced_obj<R>::value, R> api_name(              \
      Args... params) {                                                        \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::push_back_msg_cache(                                               \
        this_obj_name + "." + __FUNCTION__ + "(");                             \
    Tracer::clear_params_log_cache();                                          \
    boost::hana::tuple<Args...> params_tuple = {params...};                    \
    boost::hana::for_each(params_tuple, [&] (auto x) {                         \
      Tracer::logging_param(x, -1);                                            \
    });                                                                        \
    INVOKE_LAMBDAS_(__VA_ARGS__)                                               \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    auto api_impl = [&] (auto&&... ts) {                                       \
      return &(impl_->api_name(ts...));                                        \
    };                                                                         \
    auto proc_param_lambda = [] (auto&& t) {                                   \
      return Tracer::proc_param(t);                                            \
    };                                                                         \
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);\
    auto ret_impl_p = boost::hana::unpack(params_impl, api_impl);              \
    return *static_cast<std::decay_t<R>*>(Tracer::get_traced_obj(ret_impl_p)); \
  }                                                                            \
                                                                               \
  template <class R = Ret, class... Args>                                      \
  typename std::enable_if_t<is_traced_obj_ptr<R>::value, R> api_name(          \
      Args... params) {                                                        \
    std::string this_obj_name = TraceGetObjName();                             \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(                      \
            boost::typeindex::type_id<typename R::element_type>()              \
            .pretty_name()));                                                  \
    Tracer::push_back_msg_cache(                                               \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    Tracer::clear_params_log_cache();                                          \
    boost::hana::tuple<Args...> params_tuple = {params...};                    \
    boost::hana::for_each(params_tuple, [&] (auto x) {                         \
      Tracer::logging_param(x, -1);                                            \
    });                                                                        \
    INVOKE_LAMBDAS_(__VA_ARGS__)                                               \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    auto api_impl = [&] (auto&&... ts) {                                       \
      return impl_->api_name(ts...);                                           \
    };                                                                         \
    auto proc_param_lambda = [] (auto&& t) {                                   \
      return Tracer::proc_param(t);                                            \
    };                                                                         \
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);\
    auto obj_impl_sp = boost::hana::unpack(params_impl, api_impl);             \
    R obj_sp(obj_impl_sp);                                                     \
    Tracer::insert_obj_name(static_cast<void*>(obj_impl_sp.get()), obj_name);  \
    return obj_sp;                                                             \
  }                                                                            \
                                                                               \
  template <class R = Ret, class... Args>                                      \
  typename std::enable_if_t<is_traced_obj_ptr_vector<R>::value, R> api_name(   \
      Args... params) {                                                        \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::push_back_msg_cache(                                               \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    Tracer::clear_params_log_cache();                                          \
    boost::hana::tuple<Args...> params_tuple = {params...};                    \
    boost::hana::for_each(params_tuple, [&] (auto x) {                         \
      Tracer::logging_param(x, -1);                                            \
    });                                                                        \
    INVOKE_LAMBDAS_(__VA_ARGS__)                                               \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    auto api_impl = [&] (auto&&... ts) {                                       \
      return impl_->api_name(ts...);                                           \
    };                                                                         \
    auto proc_param_lambda = [] (auto&& t) {                                   \
      return Tracer::proc_param(t);                                            \
    };                                                                         \
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);\
    auto ret_impl = boost::hana::unpack(params_impl, api_impl);                \
    R ret;                                                                     \
    for(auto& x : ret_impl) {                                                  \
      ret.push_back(std::make_shared<typename R::value_type::element_type>(    \
          *static_cast<typename R::value_type::element_type*>(                 \
              Tracer::get_traced_obj(x.get()))));                              \
    }                                                                          \
    return ret;                                                                \
  }                                                                            \
                                                                               \
  template <class R = Ret, class... Args>                                      \
  typename std::enable_if_t<is_not_traced_obj_like<R>::value, R> api_name(     \
      Args... params) {                                                        \
    std::string this_obj_name = TraceGetObjName();                             \
    Tracer::push_back_msg_cache(                                               \
        this_obj_name + "->" + __FUNCTION__ + "(");                            \
    Tracer::clear_params_log_cache();                                          \
    boost::hana::tuple<Args...> params_tuple = {params...};                    \
    boost::hana::for_each(params_tuple, [&] (auto x) {                         \
      Tracer::logging_param(x, -1);                                            \
    });                                                                        \
    INVOKE_LAMBDAS_(__VA_ARGS__)                                               \
    Tracer::pop_params_log_cache();                                            \
    Tracer::amend_last_msg_cache(");\n");                                      \
    Tracer::msg_cache_sync_to_file();                                          \
    auto api_impl = [&] (auto&&... ts) {                                       \
      return impl_->api_name(ts...);                                           \
    };                                                                         \
    auto proc_param_lambda = [] (auto&& t) {                                   \
      return Tracer::proc_param(t);                                            \
    };                                                                         \
    auto params_impl = boost::hana::transform(params_tuple, proc_param_lambda);\
    return boost::hana::unpack(params_impl, api_impl);                         \
  }

#define GET_MACRO_OVERLOAD_4_(_1, _2, _3, _4, MACRO, ...) MACRO
#define GET_MACRO_OVERLOAD_3_(_1, _2, _3, MACRO, ...) MACRO

#define DEF_MEMFN_SP(...)                                                      \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        DEF_MEMFN_SP_4_,                                       \
                        DEF_MEMFN_SP_3_,                                       \
                        DEF_MEMFN_SP_2_)(__VA_ARGS__)

#define DEF_MEMFN(...)                                                         \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        DEF_MEMFN_4_,                                          \
                        DEF_MEMFN_3_,                                          \
                        DEF_MEMFN_2_)(__VA_ARGS__)

#define DEF_INPLACE_MEMFN(...)                                                 \
  GET_MACRO_OVERLOAD_4_(__VA_ARGS__,                                           \
                        DEF_INPLACE_MEMFN_4_,                                  \
                        DEF_INPLACE_MEMFN_3_,                                  \
                        DEF_INPLACE_MEMFN_2_)(__VA_ARGS__)

#define DEF_CONSTRUCTOR(...)                                                   \
  GET_MACRO_OVERLOAD_3_(__VA_ARGS__,                                           \
                        DEF_CONSTRUCTOR_3_,                                    \
                        DEF_CONSTRUCTOR_2_,                                    \
                        DEF_CONSTRUCTOR_1_)(__VA_ARGS__)

#define LOGGING_POINTER_MSG(offset, length, idx)                               \
  char log_msg[1024] = {0};                                                    \
  snprintf(log_msg, 1024,                                                      \
           "trace::Replayer::get_vector<char>(%u, %u).data()",                 \
           offset, length);                                                    \
  Tracer::insert_params_log_cache(std::string(log_msg), idx);

#define DEF_INTERFACE_CONSTRUCTOR(interface)                                   \
  interface(const std::shared_ptr<target::interface>& impl) {                  \
    Tracer::insert_traced_obj(static_cast<void*>(impl.get()),                  \
                              static_cast<void*>(this));                       \
    impl_ = impl;                                                              \
  }

#endif // TIM_EXPERIMENTAL_TRACE_TRACER_H_
