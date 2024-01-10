/****************************************************************************
*
*    Copyright (c) 2020-2024 Vivante Corporation
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

#ifndef VSI_NBG_RUNNER_VX_OVX_EXECUTOR_HPP_
#define VSI_NBG_RUNNER_VX_OVX_EXECUTOR_HPP_

#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_nn.h>
#include <VX/vx_types.h>

#include <array>
#include <filesystem>
#include <vector>


namespace vsi::nbg_runner::vx {

namespace fs = std::filesystem;

struct OVXTensorInfo {
  static constexpr size_t kMaxRank = 6;

  size_t rank;
  std::array<size_t, kMaxRank> shape;
  vx_enum data_type;
  vx_enum quant_type;
  vx_tensor_quant_param quant_param;
};

class OVXExecutor {
 public:
  explicit OVXExecutor(const char* nbg_data, size_t nbg_size);
  explicit OVXExecutor(const fs::path& nbg_path);

  ~OVXExecutor();

  /** \brief Class initialization. */
  int init();

  /** \brief I/O params query getters. */
  [[nodiscard]] size_t get_num_inputs() const {
    return input_tensors_infos_.size();
  }
  [[nodiscard]] size_t get_num_outputs() const {
    return output_tensors_infos_.size();
  }
  [[nodiscard]] OVXTensorInfo get_input_info(size_t index) const {
    return input_tensors_infos_[index];
  }
  [[nodiscard]] OVXTensorInfo get_output_info(size_t index) const {
    return output_tensors_infos_[index];
  }

  /** \brief Copy I/O tensor data. */
  int copy_to_input(size_t index, void* data, size_t rank, const size_t* shape,
                const size_t* strides);
  int copy_from_output(size_t index, void* data, size_t rank, const size_t* shape,
                const size_t* strides);

  int run();

 private:
  int query_nbg_io_infos();

  /** \brief The OpenVX context for management of all OpenVX objects. */
  vx_context context_;
  /** \brief The OpenVX graph for execution. */
  vx_graph graph_;
  /** \brief The OpenVX NBG node. */
  vx_node nbg_node_;
  /** \brief The OpenVX NBG kernel. */
  vx_kernel nbg_kernel_;
  /** \brief The OpenVX input tensors params. */
  std::vector<OVXTensorInfo> input_tensors_infos_;
  /** \brief The OpenVX output tensors params. */
  std::vector<OVXTensorInfo> output_tensors_infos_;
  /** \brief The OpenVX input tensors. */
  std::vector<vx_tensor> input_tensors_;
  /** \brief The OpenVX output tensors. */
  std::vector<vx_tensor> output_tensors_;

  /** \brief The NBG buffer. */
  std::vector<char> nbg_buffer_;
};

}  // namespace vsi::nbg_runner::vx

#endif