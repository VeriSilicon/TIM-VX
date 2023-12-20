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
#ifndef TIM_EXPERIMENTAL_TRACE_REPLAYER_H_
#define TIM_EXPERIMENTAL_TRACE_REPLAYER_H_

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

#define TRACE_LOG_FILE_REPLAY_ "trace_log.rpl.cc"
#define TRACE_BIN_FILE_REPLAY_ "trace_bin.rpl.bin"
#define TRACE_PREFIX_ENV_VAR_ "TRACE_DUMP_PREFIX" 

#define TCLOGE(fmt, ...) do {                                                  \
    printf("[ERROR] [%s:%s:%d]" fmt, __FILE__, __FUNCTION__, __LINE__,         \
        ##__VA_ARGS__);                                                        \
    fflush(stdout);                                                            \
  } while (0)

/*************************** definition of replayer ***************************/
namespace trace {

class Replayer {
  static FILE* file_trace_bin_;
  static FILE* open_file(const char* file_name);
 public:
  template <class T>
  static std::vector<T> get_vector(uint32_t offset, size_t vec_size);

  template <class T>
  static std::vector<T> get_vector_from(
      const char* file_path, uint32_t offset, size_t vec_size);

  template <class T, std::size_t N>
  static std::array<T, N> get_array(uint32_t offset, size_t vec_size);
};

// #define API_REPLAYER_IMPLEMENTATION
#ifdef API_REPLAYER_IMPLEMENTATION
FILE* Replayer::file_trace_bin_ = Replayer::open_file(TRACE_BIN_FILE_REPLAY_);
/* static */ FILE* Replayer::open_file(const char* file_name) {
  char* prefix = getenv(TRACE_PREFIX_ENV_VAR_);
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
      TCLOGE("Read binary data failed!\n");
    }
    delete[] buffer;
  }
  return ret_vec;
}

template <class T, std::size_t N>
/* static */ std::array<T, N> Replayer::get_array(
    uint32_t offset, size_t vec_size) {
  std::vector<T> ret_vec = get_vector<T>(offset, vec_size);
  std::array<T, N> ret_arr;
  std::copy_n(ret_vec.begin(), ret_vec.size(), ret_arr.begin());
  return ret_arr;
}

template <class T>
/* static */ std::vector<T> Replayer::get_vector_from(
    const char* file_path,uint32_t offset, size_t vec_size) {
  FILE* external_file = fopen(file_path, "r");
  if (!external_file) {
    TCLOGE("Can not open file at: %s\n", file_path);
  }
  std::vector<T> ret_vec;
  T* buffer = new T[vec_size];
  fseek(external_file, offset, SEEK_SET);
  if (fread(buffer, sizeof(T), vec_size, external_file) == vec_size) {
    ret_vec.assign(buffer, buffer + vec_size);
  } else {
    TCLOGE("Read binary data failed!\n");
  }
  delete[] buffer;
  return ret_vec;
}

#endif /* #ifdef API_TRACER_IMPLEMENTATION */

} /* namespace trace */

#endif // TIM_EXPERIMENTAL_TRACE_REPLAYER_H_
