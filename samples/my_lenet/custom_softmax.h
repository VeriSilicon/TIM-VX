/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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

#ifndef TIM_VX_OPS_CUSTOM_SOFTMAX_H_
#define TIM_VX_OPS_CUSTOM_SOFTMAX_H_

#include "tim/vx/ops/custom_base.h"

namespace tim {
namespace vx {
namespace ops {



class CustomSoftmax : public CustomOpBase {
 public:
  //scalar param for kernel function input
  using ParamTuple = std::tuple<      int,   /* sf_size */
                                      int,   /* zp_in */
                                      float   /* scale_in */                               
                                      >;
  CustomSoftmax(Graph* graph,  ParamTuple tuple_list, uint32_t input_num = 1, uint32_t output_num = 1)
      : CustomOpBase(graph, input_num, output_num, CustomSoftmax::kernel_id_,CustomSoftmax::kernel_name_)  {
    tuple_list_.swap(tuple_list);
    param_transform(tuple_list_, param_list_);

    kernel_resource_ =
        "__kernel void softmax_U8toF16_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    int sf_size,\n\
    int zp_in,\n\
    float scale_in\n\
    )\n\
{\n\
 \n\
}\n\
\n\
";
  };

 protected:
  const char* kernel_softmax = "softmax_U8toF16_2D"; 
  ParamTuple tuple_list_;
  bool trans_a_;
  bool trans_b_;
  static const char* kernel_name_;
  static int32_t kernel_id_;

  //function for setup output
  void SetupShapeInfor() override {
     outputs_size_[0].push_back(inputs_size_[0][0]);
  }

  //function for kernel select and build option
  void SetupParams(
      std::vector<tim::vx::DataType> input_types,
      std::string& build_option) override {
        if(input_types.size()==0){
          std::cout<<"something wrong"<<std::endl;
          return;
        }
        func_name_ = kernel_softmax;
        build_option = "";     
  }

  //function for kernel local size and gobal size
  void SetupEnqueue(uint32_t& dim, std::vector<size_t>& global_size,
                    std::vector<size_t>& local_size) {
    dim = 1;
    local_size[0] = 0;
    local_size[1] = 0;
    local_size[2] = 0;

    global_size[0] = 1;
    global_size[1] = 1;
    global_size[2] = 1;
  }

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override {
    return graph->CreateOperation<CustomSoftmax>(this->tuple_list_, this->input_num_, this->output_num_);
  }
};

}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif