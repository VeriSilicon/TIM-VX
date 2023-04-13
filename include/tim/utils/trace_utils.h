#ifndef TIM_VX_UTILS_TRACE_UTILS_H_
#define TIM_VX_UTILS_TRACE_UTILS_H_
#include <memory>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <list>
#include <mutex>
#include <type_traits>
#include <functional>
#include <stdio.h>
#include <stdarg.h>
#include <boost/preprocessor.hpp>
#include <boost/type_index.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/unpack.hpp>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops.h"

/************************************************************ 
Caution! Do not formatting these code with auto format tools!
*************************************************************/

/*
 * Current problems:
 * 1. There's no derive relationship between traced classes and target classes,
 *    so can't directly use the member variables form traced classes;
 * 2. The argument descriptions look ugly and fussy, and for those apis with
 *    overload, we must defines the same overloaded traced apis;
 * 3. To dump nbg buffer void*, must get length of itself, but the cal method
 *    update frequently, currently we got the length form the file length of
 *    dumped nbg file by setting "export VIV_VX_ENABLE_DUMP_NBG=1"; 
 *
 * ToDo:
 * 1. For problem 1, use SFINAE to handle template apis, then we can derive
 *    traced classes from target classes;
 * 2. For problem 2, make all traced apis as template functions, then we can
 *    define traced api like:
 *      DEF_TRACED_API(api_name, exprA, exprB)
 *    where exprX is a lambda or macro to handle void* length;
 * 3. Split traced classes to multi files;
 * 4. Review lvalue and rvalue usage;
 * 5. DEF_MEMFN_SP DEF_MEMFN and DEF_INPLACE_MEMFN can merge into only one macro
 *    by do type traits on retval;
 */

#define TRACE_LOG_NAME_ "trace_log.cc"
#define TRACE_BIN_FILE_ "trace_bin.bin"
#define REPLAY_BIN_FILE_ "trace_bin_replay.bin"
#define TRACE_PREFIX_ENV_VAR_ "VSI_TRACE_PREFIX" 

#define TCLOGE(fmt, ...)                                                       \
  printf("[ERROR] [%s:%s:%d]" fmt, __FILE__, __FUNCTION__, __LINE__,           \
         ##__VA_ARGS__)

namespace trace {
namespace target = ::tim::vx;
static const char* __trace_target_namespace_ = "tim::vx";

template<typename...>
using void_t = void;

template <class, class = void>
struct is_fundamental_vector : std::false_type {};

template <class T>
struct is_fundamental_vector<std::vector<T>> {
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
struct is_others_log_type : std::integral_constant<bool,
    !is_fundamental_vector<std::decay_t<T>>::value &&
    !std::is_enum<std::decay_t<T>>::value &&
    !std::is_fundamental<std::decay_t<T>>::value &&
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value> {};

template <class T>
struct is_others_proc_type : std::integral_constant<bool,
    !is_traced_obj<std::decay_t<T>>::value &&
    !is_traced_obj_ptr<std::decay_t<T>>::value &&
    !is_traced_obj_ptr_vector<std::decay_t<T>>::value> {};

class Replayer {
  static FILE* file_trace_bin_;

  static FILE* open_file(const char* file_name);

 public:
  template <class T>
  static std::vector<T> get_vector(uint32_t offset, size_t vec_size);
};

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
    obj_names_.insert({obj_ptr, obj_name});
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
      TCLOGE("Can't amend sub_msg, beacuse msg cache is empty!\n");
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
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  // default substitution
  template <class T,
      typename std::enable_if_t<is_others_log_type<T>::value, int> = 0>
  static inline void logging_param(const T& t, uint32_t idx) {}
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

  /*
   * template functions for pass correct parameters to api implemention
   */

  // default substitution
  template <class T,
      typename std::enable_if_t<is_others_proc_type<T>::value, int> = 0>
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

template <class TargetClass>
struct TraceClassBase {
  std::shared_ptr<TargetClass> impl_;
  TargetClass& TraceGetImpl() const { return *impl_; }
  // temporary return rvalue to prevent compile error
  std::shared_ptr<TargetClass> TraceGetImplSp() { return impl_; }
  std::string& TraceGetObjName() const {
    return Tracer::get_obj_name(static_cast<const void*>(this));
  }
};

}  // namespace trace


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

#define TO_VARIDIC_IMPL_COMMA_(r, _, elem) elem,
#define TO_VARIDIC_IMPL_NO_COMMA_(elem) elem

#define SEQ_TO_VARIDICS(seqs)                                                  \
  BOOST_PP_SEQ_FOR_EACH(TO_VARIDIC_IMPL_COMMA_, _,                             \
      BOOST_PP_SEQ_SUBSEQ(seqs, 0, BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(seqs))))     \
  TO_VARIDIC_IMPL_NO_COMMA_(                                                   \
      BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_PP_SEQ_SIZE(seqs)), seqs))

#define DEF_MEMFN_SP_2_(ret_class, api_name)                                   \
  std::shared_ptr<ret_class> api_name() {                                      \
    std::string this_obj_name = TraceGetObjName();                             \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#ret_class));         \
    Tracer::logging_msg("auto %s = %s->%s();\n", obj_name.c_str(),             \
                              this_obj_name.c_str(), __FUNCTION__);            \
    auto obj = std::make_shared<ret_class>(impl_->api_name());                 \
    Tracer::insert_obj_name(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
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
    auto obj = std::make_shared<ret_class>(                                    \
        impl_->api_name(PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))));         \
    Tracer::insert_obj_name(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
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
    auto obj = std::make_shared<ret_class>(                                    \
        impl_->api_name(PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))));         \
    Tracer::insert_obj_name(static_cast<void*>(obj.get()), obj_name);          \
    return obj;                                                                \
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
                              this_obj_name.c_str(), __FUNCTION__);            \
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
        __trace_target_namespace_, __FUNCTION__);                              \
    impl_ = std::make_shared<target::class_name>();                            \
    Tracer::insert_traced_obj(                                                 \
        static_cast<void*>(impl_.get()), static_cast<void*>(this));            \
    Tracer::insert_obj_name(static_cast<void*>(this), obj_name);               \
  }

#define DEF_CONSTRUCTOR_2_(class_name, args_desc)                              \
  class_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                            \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#class_name));        \
    Tracer::logging_msg("auto %s = %s::%s(", obj_name.c_str(),                 \
        __trace_target_namespace_, __FUNCTION__);                              \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    Tracer::dump_params_log_cache();                                           \
    Tracer::logging_msg(");\n");                                               \
    impl_ = std::make_shared<target::class_name>(                              \
        PROC_PARAMS(ARGS_DESC_TO_PARAMS(args_desc)));                          \
    Tracer::insert_traced_obj(                                                 \
        static_cast<void*>(impl_.get()), static_cast<void*>(this));            \
    Tracer::insert_obj_name(static_cast<void*>(this), obj_name);               \
  }

#define DEF_CONSTRUCTOR_3_(class_name, args_desc, SPECIAL_MACRO_)              \
  class_name(ARGS_DESC_TO_DECLARATION(args_desc)) {                            \
    std::string obj_name =                                                     \
        Tracer::allocate_obj_name(Tracer::get_obj_prefix(#class_name));        \
    Tracer::push_back_msg_cache(                                               \
        "auto " + obj_name + " = " + __trace_target_namespace_ + "::" +        \
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
    Tracer::insert_obj_name(static_cast<void*>(this), obj_name);               \
  }

#define SPECIALIZATION_CREATE_OP_1_(opname)                                    \
template <class... Params>                                                     \
inline std::shared_ptr<trace::ops::opname> Graph::CreateOperationImpl(         \
      ops::_VSI_Tag_of_ ## opname, Params... params) {                         \
  std::string this_obj_name = TraceGetObjName();                               \
  std::string obj_name = Tracer::allocate_obj_name(std::string(#opname) + "_");\
  Tracer::logging_msg(                                                         \
      "auto %s = %s->CreateOperation<%s::ops::%s>(", obj_name.c_str(),         \
      this_obj_name.c_str(), __trace_target_namespace_, #opname);              \
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
  auto op = std::make_shared<trace::ops::opname>(                              \
      boost::hana::unpack(params_impl, CreateOpImpl));                         \
  Tracer::insert_obj_name(static_cast<void*>(op.get()), obj_name);             \
  return op;                                                                   \
}

#define SPECIALIZATION_CREATE_OP_2_(opname, args_desc)                         \
template <>                                                                    \
std::shared_ptr<trace::ops::opname> Graph::CreateOperation(                    \
    ARGS_DESC_TO_DECLARATION(args_desc)) {                                     \
  std::string this_obj_name = TraceGetObjName();                               \
  std::string obj_name = Tracer::allocate_obj_name(std::string(#opname) + "_");\
  Tracer::logging_msg(                                                         \
      "auto %s = %s->CreateOperation<target::ops::%s>(",                       \
      obj_name.c_str(), this_obj_name.c_str(), #opname);                       \
    LOG_PARAMS(ARGS_DESC_TO_PARAMS(args_desc))                                 \
    Tracer::dump_params_log_cache();                                           \
    Tracer::logging_msg(");\n");                                               \
  auto op = std::make_shared<trace::ops::opname>(                              \
      impl_->CreateOperation<target::ops::opname>(                             \
          SEQ_TO_VARIDICS(ARGS_DESC_TO_PARAMS(args_desc))));                   \
  Tracer::insert_obj_name(static_cast<void*>(op.get()), obj_name);             \
  return op;                                                                   \
}

#define SPECIALIZATION_CREATE_OP_3_(opname, args_desc, SPECIAL_MACRO_)         \
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
  auto op = std::make_shared<trace::ops::opname>(                              \
      impl_->CreateOperation<target::ops::opname>(                             \
          SEQ_TO_VARIDICS(ARGS_DESC_TO_PARAMS(args_desc))));                   \
  Tracer::insert_obj_name(static_cast<void*>(op.get()), obj_name);             \
  return op;                                                                   \
}

#define LOGGING_PONITER_MSG(offset, length, idx)                               \
  char log_msg[1024] = {0};                                                    \
  snprintf(log_msg, 1024,                                                      \
           "trace::Replayer::get_vector<char>(%u, %u).data()",                 \
           offset, length);                                                    \
  Tracer::insert_params_log_cache(std::string(log_msg), idx);

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

#define VSI_SPECIALIZATION_CREATE_OP(...)                                      \
  GET_MACRO_OVERLOAD_3_(__VA_ARGS__,                                           \
                        SPECIALIZATION_CREATE_OP_3_,                           \
                        SPECIALIZATION_CREATE_OP_2_,                           \
                        SPECIALIZATION_CREATE_OP_1_)(__VA_ARGS__)

namespace trace {
using ShapeType = std::vector<uint32_t>;

struct Quantization : public TraceClassBase<target::Quantization> {
  DEF_CONSTRUCTOR(Quantization)
};

struct TensorSpec : public TraceClassBase<target::TensorSpec> {
  DEF_CONSTRUCTOR(TensorSpec)
  DEF_CONSTRUCTOR(TensorSpec,
                  ((target::DataType))((const ShapeType&))
                      ((target::TensorAttribute)))

  DEF_CONSTRUCTOR(TensorSpec,
                  ((target::DataType))((const ShapeType&))
                      ((target::TensorAttribute))((const Quantization&)))
  DEF_INPLACE_MEMFN(TensorSpec&,
                    SetAttribute,
                    ((target::TensorAttribute)))
};

} // namespace trace

namespace trace {
struct Tensor : public TraceClassBase<target::Tensor> {
  Tensor(const std::shared_ptr<target::Tensor>& impl) {
    Tracer::insert_traced_obj(static_cast<void*>(impl.get()),
                              static_cast<void*>(this));
    impl_ = impl;
  }

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t data_length = BOOST_PP_SEQ_ELEM(1, params);                         \
  uint32_t offset =                                                            \
      Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params),                          \
                             sizeof(char), data_length);                       \
  LOGGING_PONITER_MSG(offset, data_length, 0)

  // arguments description must format as: ((dtype)) or ((dtype)(default_value))
  DEF_MEMFN(bool,
            CopyDataToTensor,
            ((const void*))((uint32_t)(0)),
            SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t data_length = impl_->GetSpec().GetByteSize();                       \
  uint32_t offset =                                                            \
      Tracer::dump_data(BOOST_PP_SEQ_ELEM(0, params),                          \
                             sizeof(char), data_length);                       \
  LOGGING_PONITER_MSG(offset, data_length, 0)

  DEF_MEMFN(bool,
            CopyDataFromTensor,
            ((void*)),
            SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

  DEF_MEMFN(uint32_t,
            GetId)
  TensorSpec& GetSpec() {
    std::string this_obj_name = TraceGetObjName();
    Tracer::logging_msg("%s->%s();\n",
    this_obj_name.c_str(), __FUNCTION__);
    return *static_cast<TensorSpec*>(Tracer::get_traced_obj(&impl_->GetSpec()));
  }
  DEF_MEMFN(const ShapeType&,
            GetShape)
  DEF_MEMFN(bool,
            IsConstTensor)
};


} // namespace trace

namespace trace {

struct Operation : public TraceClassBase<target::Operation> {
  Operation(const std::shared_ptr<target::Operation>& impl) { impl_ = impl; }

  DEF_INPLACE_MEMFN(Operation&,
                    BindInput,
                    ((const std::shared_ptr<Tensor>&)))

  DEF_INPLACE_MEMFN(Operation&,
                    BindOutput,
                    ((const std::shared_ptr<Tensor>&)))

#define SPECIAL_MACRO_(params)                                                 \
  uint32_t vec_size = BOOST_PP_SEQ_ELEM(0, params).size();                     \
  Tracer::amend_last_msg_cache("{");                                           \
  for (uint32_t i = 0; i < vec_size - 1; i++) {                                \
    Tracer::amend_last_msg_cache(                                              \
        BOOST_PP_SEQ_ELEM(0, params)[i]->TraceGetObjName() + ", ");            \
  }                                                                            \
  Tracer::amend_last_msg_cache(                                                \
      BOOST_PP_SEQ_ELEM(0, params).back()->TraceGetObjName());                 \
  Tracer::amend_last_msg_cache("}");

  DEF_INPLACE_MEMFN(Operation&,
                    BindInputs,
                    ((const std::vector<std::shared_ptr<Tensor>>&)),
                    SPECIAL_MACRO_)

  DEF_INPLACE_MEMFN(Operation&,
                    BindOutputs,
                    ((const std::vector<std::shared_ptr<Tensor>>&)),
                    SPECIAL_MACRO_)

#undef SPECIAL_MACRO_
};

} // namespace trace

namespace trace {
namespace ops {

struct DefaultTag {};
template<class T>
struct TagDispatchTrait {
  using tag = DefaultTag;
};

#define DEF_TIMVX_OP_IMPL_(r, _, op)                                           \
struct op : Operation {                                                        \
  op(const std::shared_ptr<target::ops::op>& impl) : Operation(impl) {}        \
};                                                                             \
struct BOOST_PP_CAT(_VSI_Tag_of_, op) {};                                      \
template<>                                                                     \
struct TagDispatchTrait<op> {                                                  \
  using tag = BOOST_PP_CAT(_VSI_Tag_of_, op);                                  \
};


#define DEF_TIMVX_OPS_AND_TAGS(ops)                                            \
  BOOST_PP_SEQ_FOR_EACH(DEF_TIMVX_OP_IMPL_, _, ops)

DEF_TIMVX_OPS_AND_TAGS(
  (Add)
  (Broadcast)
  (Div)
  (Cast)
  (Select)
  (Multiply)
  (Sub)
  (Sqrt)
  (Exp)
  (GreaterOrEqual)
  (Rsqrt)
  (Equal)
  (Transpose)
  (Maximum)
  (Matmul)
  (ReduceSum)
  (Reshape)
  (Pow)
  (NBG)
)

} // namespace ops
} // namespace trace

namespace trace {

#define DECL_CREATE_OP_IMPL(op)                                                \
  template <class... Params>                                                   \
  inline std::shared_ptr<trace::ops::op> CreateOperationImpl(                  \
      ops::_VSI_Tag_of_ ## op, Params... params);                              \

struct Graph : public TraceClassBase<target::Graph> {
  Graph(const std::shared_ptr<target::Graph>& impl) { impl_ = impl; }

#define SPECIAL_MACRO_(params)                                                 \
  if (BOOST_PP_SEQ_ELEM(1, params) == nullptr) {                               \
    Tracer::insert_params_log_cache("nullptr", 1);                             \
  } else {                                                                     \
    uint32_t data_length =                                                     \
        BOOST_PP_SEQ_ELEM(0, params).TraceGetImpl().GetByteSize();             \
    uint32_t offset =                                                          \
        Tracer::dump_data(                                                     \
            BOOST_PP_SEQ_ELEM(1, params), sizeof(char), data_length);          \
    LOGGING_PONITER_MSG(offset, data_length, 0)                                \
  }

  DEF_MEMFN_SP(Tensor,
               CreateTensor,
               ((const TensorSpec&))((const void*)(nullptr)),
               SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

#define SPECIAL_MACRO_(params)                                                 \
  if (BOOST_PP_SEQ_ELEM(0, params) == nullptr) {                               \
    std::string size_name = Tracer::allocate_obj_name("nbg_size_");            \
    Tracer::insert_obj_name(BOOST_PP_SEQ_ELEM(1, params), size_name);          \
    Tracer::insert_before_last_msg_cache(                                      \
        "size_t " + size_name + " = -1;\n");                                   \
    Tracer::insert_params_log_cache("nullptr", 0);                             \
  } else {                                                                     \
    uint32_t data_length = *BOOST_PP_SEQ_ELEM(1, params);                      \
    uint32_t offset = Tracer::dump_data(                                       \
        BOOST_PP_SEQ_ELEM(0, params), sizeof(char), data_length);              \
    LOGGING_PONITER_MSG(offset, data_length, 0)                                \
  }                                                                            \
  Tracer::insert_params_log_cache(                                             \
      "&" + Tracer::get_obj_name(BOOST_PP_SEQ_ELEM(1, params)), 1);

  DEF_MEMFN(bool,
            CompileToBinary,
            ((void*))((size_t*)),
            SPECIAL_MACRO_)

  DEF_MEMFN(void,
            PrintGraph)

  const std::vector<std::shared_ptr<Tensor>> InputsTensor() {
    std::string this_obj_name = TraceGetObjName();
    Tracer::logging_msg("%s->%s();\n", this_obj_name.c_str(), __FUNCTION__);
    std::vector<std::shared_ptr<Tensor>> ret;
    for(auto& x : impl_->InputsTensor()) {
      ret.push_back(std::make_shared<Tensor>(
          *static_cast<Tensor*>(Tracer::get_traced_obj(x.get()))));
    }
    return ret;
  }

  const std::vector<std::shared_ptr<Tensor>> OutputsTensor() {
    std::string this_obj_name = TraceGetObjName();
    Tracer::logging_msg("%s->%s();\n", this_obj_name.c_str(), __FUNCTION__);
    std::vector<std::shared_ptr<Tensor>> ret;
    for(auto& x : impl_->OutputsTensor()) {
      ret.push_back(std::make_shared<Tensor>(
          *static_cast<Tensor*>(Tracer::get_traced_obj(x.get()))));
    }
    return ret;
  }

  // DEF_MEMFN(const std::vector<std::shared_ptr<Tensor>>,
  //               InputsTensor)

  // DEF_MEMFN(const std::vector<std::shared_ptr<Tensor>>,
  //               OutputsTensor)

#undef SPECIAL_MACRO_

  DEF_MEMFN(bool, Compile)

  DEF_MEMFN(bool, Run)

  template <class OpType, class... Params>
  std::shared_ptr<OpType> CreateOperation(Params... params) {
    return CreateOperationImpl(
      typename ops::TagDispatchTrait<OpType>::tag {}, params...);
  }

 private:
  DECL_CREATE_OP_IMPL(Add)
  DECL_CREATE_OP_IMPL(Broadcast)
  DECL_CREATE_OP_IMPL(Div)
  DECL_CREATE_OP_IMPL(Cast)
  DECL_CREATE_OP_IMPL(Select)
  DECL_CREATE_OP_IMPL(Multiply)
  DECL_CREATE_OP_IMPL(Sub)
  DECL_CREATE_OP_IMPL(Sqrt)
  DECL_CREATE_OP_IMPL(Exp)
  DECL_CREATE_OP_IMPL(GreaterOrEqual)
  DECL_CREATE_OP_IMPL(Rsqrt)
  DECL_CREATE_OP_IMPL(Equal)
  DECL_CREATE_OP_IMPL(Transpose)
  DECL_CREATE_OP_IMPL(Maximum)
  DECL_CREATE_OP_IMPL(Matmul)
  DECL_CREATE_OP_IMPL(ReduceSum)
  DECL_CREATE_OP_IMPL(Reshape)
  DECL_CREATE_OP_IMPL(Pow)
};

VSI_SPECIALIZATION_CREATE_OP(Add)
VSI_SPECIALIZATION_CREATE_OP(Broadcast)
VSI_SPECIALIZATION_CREATE_OP(Div)
VSI_SPECIALIZATION_CREATE_OP(Cast)
VSI_SPECIALIZATION_CREATE_OP(Select)
VSI_SPECIALIZATION_CREATE_OP(Multiply)
VSI_SPECIALIZATION_CREATE_OP(Sub)
VSI_SPECIALIZATION_CREATE_OP(Sqrt)
VSI_SPECIALIZATION_CREATE_OP(Exp)
VSI_SPECIALIZATION_CREATE_OP(GreaterOrEqual)
VSI_SPECIALIZATION_CREATE_OP(Rsqrt)
VSI_SPECIALIZATION_CREATE_OP(Equal)
VSI_SPECIALIZATION_CREATE_OP(Transpose)
VSI_SPECIALIZATION_CREATE_OP(Maximum)
VSI_SPECIALIZATION_CREATE_OP(Matmul)
VSI_SPECIALIZATION_CREATE_OP(ReduceSum)
VSI_SPECIALIZATION_CREATE_OP(Reshape)
VSI_SPECIALIZATION_CREATE_OP(Pow)

#define SPECIAL_MACRO_(params)                                                 \
  std::string buf_name = Tracer::allocate_obj_name("nbg_buf_vec_");            \
  FILE* nbg_dumped = fopen("network_binary_graph.nb", "r");                    \
  fseek(nbg_dumped, 0L, SEEK_END);                                             \
  uint32_t data_length = ftell(nbg_dumped);                                    \
  fclose(nbg_dumped);                                                          \
  uint32_t offset = Tracer::dump_data(                                         \
      BOOST_PP_SEQ_ELEM(0, params), sizeof(char), data_length);                \
  Tracer::insert_before_last_msg_cache("std::vector<char> " + buf_name +       \
      " = trace::Replayer::get_vector<char>(" + std::to_string(offset)  +      \
      "," + std::to_string(data_length) + ");\n");                             \
  Tracer::insert_params_log_cache(buf_name + ".data()", 0);

VSI_SPECIALIZATION_CREATE_OP(NBG,
                             ((const char*))((size_t))((size_t)),
                             SPECIAL_MACRO_)

#undef SPECIAL_MACRO_

}  // namespace trace

namespace trace {

struct Context : public TraceClassBase<target::Context> {
  Context(const std::shared_ptr<target::Context>& impl) { impl_ = impl; }

  static std::shared_ptr<Context> Create();

  DEF_MEMFN_SP(Graph, CreateGraph)
};


}  // namespace trace
#endif  // TIM_VX_UTILS_TRACE_UTILS_H_
