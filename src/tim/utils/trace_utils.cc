#include "tim/utils/trace_utils.h"

namespace trace {
  FILE* Replayer::file_trace_bin_ = Replayer::open_file(REPLAY_BIN_FILE_);

  /* static */ FILE* Replayer::open_file(const char* file_name) {
    char* prefix = getenv("VSI_TRACE_PREFIX");
    FILE* fp;
    char path[1024] = {0};
    if (prefix != NULL) {
      strcpy(path, prefix);
      strcat(path, file_name);
    } else {
      strcpy(path, file_name);
    }
    fp = fopen(path, "r");
    if (!fp) {
      TCLOGE("Can not open file at: %s\n", path);
    }
    return fp;
  }

  template <class T>
  /* static */ std::vector<T> Replayer::get_vector(
      uint32_t offset, size_t vec_size) {
    std::vector<T> ret_vec;
    if (!file_trace_bin_) {
      TCLOGE("FILE pointer is NULL!\n");
    } else {
      T* buffer = new T[vec_size];
      fseek(file_trace_bin_, offset, SEEK_SET);
      if (fread(buffer, sizeof(T), vec_size, file_trace_bin_) == vec_size) {
        ret_vec.assign(buffer, buffer + vec_size);
      } else {
        TCLOGE("Read bin data failed!\n");
      }
      delete[] buffer;
    }
    return ret_vec;
  }

} // namespace trace

namespace trace {
  std::unordered_map<const void*, std::string> Tracer::obj_names_;
  std::vector<std::string> Tracer::params_log_cache_;
  std::list<std::string> Tracer::msg_cache_;
  std::unordered_map<const void*, void*> Tracer::target2trace_map_;
  FILE* Tracer::file_trace_log_ = Tracer::open_file(TRACE_LOG_NAME_);
  FILE* Tracer::file_trace_bin_ = Tracer::open_file(TRACE_BIN_FILE_);
  std::unordered_map<std::string, std::string> Tracer::objs_prefix_ = {
    {"Quantization", "quant_"    },
    {"TensorSpec",  "spec_"      },
    {"Tensor",      "tensor_"    },
    {"Graph",       "graph_"     }
  };

  std::mutex Tracer::log_mtx_;
  std::mutex Tracer::bin_mtx_;

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
      TCLOGE("log file do not exist\n");
    }
    std::lock_guard<std::mutex> lock(log_mtx_);
    fprintf(file_trace_log_, "%s", arg_buffer);
    // fflush(file_trace_log_);
  }

  /* static */ uint32_t Tracer::dump_data(
      const void* data, size_t byte_size, size_t count) {
    std::lock_guard<std::mutex> lock(bin_mtx_);
    if (fwrite(data, byte_size, count, file_trace_bin_) != count) {
      TCLOGE("Write trace binary data failed!\n");
    }
    static uint32_t offset = 0;
    uint32_t temp = offset;
    offset += byte_size * count;
    return temp;
  }


std::shared_ptr<Context> Context::Create() {
  std::string obj_name = Tracer::allocate_obj_name("ctx_");
  std::string pf(__PRETTY_FUNCTION__);
  pf.replace(pf.rfind("trace"), 5, __trace_target_namespace_);
  char log_msg[1024] = {0};
  snprintf(log_msg, 1024, "auto %s =%s;\n", obj_name.c_str(),
           pf.substr(pf.rfind(" "), pf.size()).c_str());
  Tracer::logging_msg(log_msg);
  auto obj = std::make_shared<Context>(target::Context::Create());
  Tracer::insert_obj_name(static_cast<void*>(obj.get()), obj_name);
  return obj;
}

} // namespace trace
