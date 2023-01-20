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

#ifndef TIM_VX_OPS_CUSTOM_GEMM_H_
#define TIM_VX_OPS_CUSTOM_GEMM_H_

#include "tim/vx/ops/custom_base.h"

namespace tim {
namespace vx {
namespace ops {



class CustomGemm : public CustomOpBase {
 public:
  //scalar param for kernel function input
  using ParamTuple = std::tuple<int,   /* M */
                                      int,   /* K */
                                      int,   /* N */
                                      int,   /* ac2zero */
                                      int,   /* bc2zero */
                                      float, /* scale_a */
                                      float, /* zp_a */
                                      float, /* scale_b */
                                      float, /* zp_b */
                                      float, /* scale_out */
                                      float  /* zp_out */
                                      >;
  CustomGemm(Graph* graph, bool trans_a, bool trans_b,
             ParamTuple tuple_list, uint32_t input_num = 2,
             uint32_t output_num = 1)
      : CustomOpBase(graph, input_num, output_num, CustomGemm::kernel_id_,
                     CustomGemm::kernel_name_),
        trans_a_(trans_a),
        trans_b_(trans_b) {
    tuple_list_.swap(tuple_list);
    param_transform(tuple_list_, param_list_);

    kernel_resource_ =
        "__kernel void gemm_F32F32toF32_2D(\n\
    __read_only image2d_t   inputA,\n\
    __read_only image2d_t   inputB,\n\
    __write_only image2d_t  output,\n\
    int M,\n\
    int K,\n\
    int N,\n\
    int ac2zero,\n\
    int bc2zero,\n\
    float scale_a,\n\
    float zp_a,\n\
    float scale_b,\n\
    float zp_b,\n\
    float scale_out,\n\
    float zp_out\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    float4 sum = (float4)(0);\n\
\n\
    for(; coord.z < K;)\n\
    {\n\
        float4 tempA0;\n\
        float4 tempB0;\n\
\n\
        tempA0 = read_imagef(inputA, coord.zy);\n\
        tempB0 = read_imagef(inputB, coord.xz);\n\
        coord.z++;\n\
\n\
        sum = sum + tempA0 * tempB0;\n\
    }\n\
    write_imagef(output, coord.xy, sum);\n\
}\n\
\n\
";
  };

 protected:
  const char* kernel_NotTransA_NotTransB = "gemm_F32F32toF32_2D";
  const char* kernel_TransA_NotTransB = ".....";
  ParamTuple tuple_list_;
  bool trans_a_;
  bool trans_b_;
  static const char* kernel_name_;
  static int32_t kernel_id_;

  //function for setup output
  void SetupShapeInfor() override {
    if (!trans_a_ && !trans_a_) {
      outputs_size_[0].push_back(inputs_size_[0][1]);
      outputs_size_[0].push_back(inputs_size_[1][0]);
    } else {
      //other situation: set up outputs_size
      //outputs_size_[0].push_back()......
    }
  }

  //function for kernel select and build option
  void SetupParams(
      std::vector<tim::vx::DataType> input_types,
      std::string& build_option) override {
    if (trans_a_ == false &&
        trans_a_ == false &&
        input_types[0] == tim::vx::DataType::FLOAT32 &&
        input_types[1] == tim::vx::DataType::FLOAT32) {
      func_name_ = kernel_NotTransA_NotTransB;
      build_option = "";
    } else {
      // other situation: named func_name_ and setup param_list
    }
  }

  //function for kernel local size and gobal size
  void SetupEnqueue(uint32_t& dim, std::vector<size_t>& global_size,
                    std::vector<size_t>& local_size) {
    dim = 3;
    local_size[0] = 0;
    local_size[1] = 0;
    local_size[2] = 0;

    global_size[0] = gpu_align(outputs_size_[0][0], 4);
    global_size[1] = gpu_align(outputs_size_[0][1], 4);
    global_size[2] = outputs_size_[0].size() > 2 ? outputs_size_[0][2] : 1;
  }

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override {
    return graph->CreateOperation<CustomGemm>(trans_a_,trans_b_,
        this->tuple_list_, this->input_num_, this->output_num_);
  }
};

}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif