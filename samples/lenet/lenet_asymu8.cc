/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "lenet_asymu8_weights.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/softmax.h"
#include "tim/vx/tensor.h"

std::vector<uint8_t> input_data = {
    0,   0,   0,   0,   0,   0,   0,   0,   6,   0,   2,   0,   0,   8,   0,
    3,   0,   7,   0,   2,   0,   0,   0,   10,  0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   3,   1,   1,   0,   14,  0,   0,   3,   0,
    2,   4,   0,   0,   0,   3,   1,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   4,   3,   0,   0,   0,   5,   0,   4,   0,   0,
    0,   0,   10,  12,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   6,   5,   0,   2,   0,   9,   0,   12,  2,   0,   5,   1,   0,
    0,   2,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    3,   0,   33,  0,   0,   155, 186, 55,  17,  22,  0,   0,   3,   9,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    2,   0,   167, 253, 255, 235, 255, 240, 134, 36,  0,   6,   1,   4,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,   6,   87,
    240, 251, 254, 254, 237, 255, 252, 191, 27,  0,   0,   5,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   19,  226, 255, 235,
    255, 255, 254, 242, 255, 255, 68,  12,  0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   4,   1,   58,  254, 255, 158, 0,   2,
    47,  173, 253, 247, 255, 65,  4,   1,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   162, 240, 248, 92,  8,   0,   13,  0,
    88,  249, 244, 148, 0,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   17,  64,  244, 255, 210, 0,   0,   1,   2,   0,   52,  223,
    255, 223, 0,   11,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   144, 245, 255, 142, 0,   4,   9,   0,   6,   0,   37,  222, 226,
    42,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   73,
    255, 243, 104, 0,   0,   0,   0,   11,  0,   0,   0,   235, 242, 101, 4,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   133, 245, 226,
    12,  4,   15,  0,   0,   0,   0,   24,  0,   235, 246, 41,  0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   236, 245, 152, 0,   10,
    0,   0,   0,   0,   6,   0,   28,  227, 239, 1,   6,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   227, 240, 53,  4,   0,   0,   24,
    0,   1,   0,   8,   181, 249, 177, 0,   2,   0,   0,   0,   0,   4,   0,
    6,   1,   5,   0,   0,   87,  246, 219, 14,  0,   0,   2,   0,   10,  7,
    0,   134, 255, 249, 104, 4,   0,   0,   0,   0,   0,   8,   0,   3,   0,
    0,   0,   4,   89,  255, 228, 0,   11,  0,   8,   14,  0,   0,   100, 250,
    248, 236, 0,   0,   8,   0,   0,   0,   0,   5,   0,   2,   0,   0,   2,
    6,   68,  250, 228, 6,   6,   0,   0,   1,   0,   140, 240, 253, 238, 51,
    31,  0,   3,   0,   0,   0,   0,   0,   0,   5,   0,   0,   2,   0,   26,
    215, 255, 119, 0,   21,  1,   40,  156, 233, 244, 239, 103, 0,   6,   6,
    0,   0,   0,   0,   0,   0,   0,   5,   0,   0,   0,   0,   0,   225, 251,
    240, 141, 118, 139, 222, 244, 255, 249, 112, 17,  0,   0,   8,   3,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   84,  245, 255, 247,
    255, 249, 255, 255, 249, 132, 11,  0,   9,   3,   1,   1,   0,   0,   0,
    0,   2,   0,   0,   1,   0,   0,   6,   1,   0,   166, 236, 255, 255, 248,
    249, 248, 72,  0,   0,   16,  0,   16,  0,   4,   0,   0,   0,   0,   0,
    0,   0,   6,   0,   0,   4,   0,   0,   20,  106, 126, 188, 190, 112, 28,
    0,   21,  0,   1,   2,   0,   0,   3,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,
};

template <typename T>
static void printTopN(const T* prob, int outputCount, int topNum) {
  std::vector<std::tuple<int, T>> data;

  for (int i = 0; i < outputCount; i++) {
    data.push_back(std::make_tuple(i, prob[i]));
  }

  std::sort(data.begin(), data.end(),
            [](auto& a, auto& b) { return std::get<1>(a) > std::get<1>(b); });

  std::cout << " --- Top" << topNum << " ---" << std::endl;
  for (int i = 0; i < topNum; i++) {
    std::cout << std::setw(3) << std::get<0>(data[i]) << ": " << std::fixed
              << std::setprecision(6) << std::get<1>(data[i]) << std::endl;
  }
}

int main(int argc, char** argv) {
  (void) argc, (void) argv;
  auto context = tim::vx::Context::Create();
  auto graph = context->CreateGraph();

  tim::vx::ShapeType input_shape({28, 28, 1, 1});
  tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 0.00390625f,
                                    0);
  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, input_quant);
  auto input = graph->CreateTensor(input_spec);

  tim::vx::ShapeType conv1_weight_shape({5, 5, 1, 20});
  tim::vx::Quantization conv1_weighteight_quant(tim::vx::QuantType::ASYMMETRIC,
                                                0.00336234f, 119);
  tim::vx::TensorSpec conv1_weighteight_spec(
      tim::vx::DataType::UINT8, conv1_weight_shape,
      tim::vx::TensorAttribute::CONSTANT, conv1_weighteight_quant);
  auto conv1_weight =
      graph->CreateTensor(conv1_weighteight_spec, &lenet_weights[0]);

  tim::vx::ShapeType conv1_bias_shape({20});
  tim::vx::Quantization conv1_bias_quant(tim::vx::QuantType::ASYMMETRIC,
                                         1.313e-05f, 0);
  tim::vx::TensorSpec conv1_bias_spec(
      tim::vx::DataType::INT32, conv1_bias_shape,
      tim::vx::TensorAttribute::CONSTANT, conv1_bias_quant);
  auto conv1_bias = graph->CreateTensor(conv1_bias_spec, &lenet_weights[500]);

  tim::vx::Quantization conv1_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.01928069f, 140);
  tim::vx::TensorSpec conv1_output_spec(tim::vx::DataType::UINT8, {},
                                        tim::vx::TensorAttribute::TRANSIENT,
                                        conv1_output_quant);
  auto conv1_output = graph->CreateTensor(conv1_output_spec);

  tim::vx::Quantization pool1_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.01928069f, 140);
  tim::vx::TensorSpec pool1_output_spec(tim::vx::DataType::UINT8, {},
                                        tim::vx::TensorAttribute::TRANSIENT,
                                        pool1_output_quant);
  auto pool1_output = graph->CreateTensor(pool1_output_spec);

  tim::vx::ShapeType conv2_weight_shape({5, 5, 20, 50});
  tim::vx::Quantization conv2_weight_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.0011482f, 128);
  tim::vx::TensorSpec conv2_weight_spec(
      tim::vx::DataType::UINT8, conv2_weight_shape,
      tim::vx::TensorAttribute::CONSTANT, conv2_weight_quant);
  auto conv2_weight =
      graph->CreateTensor(conv2_weight_spec, &lenet_weights[580]);

  tim::vx::ShapeType conv2_bias_shape({50});
  tim::vx::Quantization conv2_bias_quant(tim::vx::QuantType::ASYMMETRIC,
                                         2.214e-05f, 0);
  tim::vx::TensorSpec conv2_bias_spec(
      tim::vx::DataType::INT32, conv2_bias_shape,
      tim::vx::TensorAttribute::CONSTANT, conv2_bias_quant);
  auto conv2_bias = graph->CreateTensor(conv2_bias_spec, &lenet_weights[25580]);

  tim::vx::Quantization conv2_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.04075872f, 141);
  tim::vx::TensorSpec conv2_output_spec(tim::vx::DataType::UINT8, {},
                                        tim::vx::TensorAttribute::TRANSIENT,
                                        conv2_output_quant);
  auto conv2_output = graph->CreateTensor(conv2_output_spec);

  tim::vx::Quantization pool2_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.04075872f, 141);
  tim::vx::TensorSpec pool2_output_spec(tim::vx::DataType::UINT8, {},
                                        tim::vx::TensorAttribute::TRANSIENT,
                                        pool2_output_quant);
  auto pool2_output = graph->CreateTensor(pool2_output_spec);

  tim::vx::ShapeType fc3_weight_shape({800, 500});
  tim::vx::Quantization fc3_weight_quant(tim::vx::QuantType::ASYMMETRIC,
                                         0.00073548f, 130);
  tim::vx::TensorSpec fc3_weight_spec(
      tim::vx::DataType::UINT8, fc3_weight_shape,
      tim::vx::TensorAttribute::CONSTANT, fc3_weight_quant);
  auto fc3_weight = graph->CreateTensor(fc3_weight_spec, &lenet_weights[25780]);

  tim::vx::ShapeType fc3_bias_shape({500});
  tim::vx::Quantization fc3_bias_quant(tim::vx::QuantType::ASYMMETRIC,
                                       2.998e-05f, 0);
  tim::vx::TensorSpec fc3_bias_spec(tim::vx::DataType::INT32, fc3_bias_shape,
                                    tim::vx::TensorAttribute::CONSTANT,
                                    fc3_bias_quant);
  auto fc3_bias = graph->CreateTensor(fc3_bias_spec, &lenet_weights[425780]);

  tim::vx::Quantization fc3_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                         0.01992089f, 0);
  tim::vx::TensorSpec fc3_output_spec(tim::vx::DataType::UINT8, {},
                                      tim::vx::TensorAttribute::TRANSIENT,
                                      fc3_output_quant);
  auto fc3_output = graph->CreateTensor(fc3_output_spec);

  tim::vx::Quantization relu_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                          0.01992089f, 0);
  tim::vx::TensorSpec relu_output_spec(tim::vx::DataType::UINT8, {},
                                       tim::vx::TensorAttribute::TRANSIENT,
                                       relu_output_quant);
  auto relu_output = graph->CreateTensor(relu_output_spec);

  tim::vx::ShapeType fc4_weight_shape({500, 10});
  tim::vx::Quantization fc4_weight_quant(tim::vx::QuantType::ASYMMETRIC,
                                         0.00158043f, 135);
  tim::vx::TensorSpec fc4_weight_spec(
      tim::vx::DataType::UINT8, fc4_weight_shape,
      tim::vx::TensorAttribute::CONSTANT, fc4_weight_quant);
  auto fc4_weight =
      graph->CreateTensor(fc4_weight_spec, &lenet_weights[427780]);

  tim::vx::ShapeType fc4_bias_shape({10});
  tim::vx::Quantization fc4_bias_quant(tim::vx::QuantType::ASYMMETRIC,
                                       3.148e-05f, 0);
  tim::vx::TensorSpec fc4_bias_spec(tim::vx::DataType::INT32, fc4_bias_shape,
                                    tim::vx::TensorAttribute::CONSTANT,
                                    fc4_bias_quant);
  auto fc4_bias = graph->CreateTensor(fc4_bias_spec, &lenet_weights[432780]);

  tim::vx::Quantization fc4_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                         0.06251489f, 80);
  tim::vx::TensorSpec fc4_output_spec(tim::vx::DataType::UINT8, {},
                                      tim::vx::TensorAttribute::TRANSIENT,
                                      fc4_output_quant);
  auto fc4_output = graph->CreateTensor(fc4_output_spec);

  tim::vx::ShapeType output_shape({10, 1});
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output = graph->CreateTensor(output_spec);

  auto conv1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      conv1_weight_shape[3], tim::vx::PadType::VALID,
      std::array<uint32_t, 2>({5, 5}), std::array<uint32_t, 2>({1, 1}),
      std::array<uint32_t, 2>({1, 1}));
  (*conv1)
      .BindInputs({input, conv1_weight, conv1_bias})
      .BindOutputs({conv1_output});

  auto pool1 = graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::MAX, tim::vx::PadType::NONE,
      std::array<uint32_t, 2>({2, 2}), std::array<uint32_t, 2>({2, 2}));
  (*pool1).BindInputs({conv1_output}).BindOutputs({pool1_output});

  auto conv2 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      conv2_weight_shape[3], tim::vx::PadType::VALID,
      std::array<uint32_t, 2>({5, 5}), std::array<uint32_t, 2>({1, 1}),
      std::array<uint32_t, 2>({1, 1}));
  (*conv2)
      .BindInputs({pool1_output, conv2_weight, conv2_bias})
      .BindOutputs({conv2_output});

  auto pool2 = graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::MAX, tim::vx::PadType::NONE,
      std::array<uint32_t, 2>({2, 2}), std::array<uint32_t, 2>({2, 2}));
  (*pool2).BindInputs({conv2_output}).BindOutputs({pool2_output});

  auto fc3 = graph->CreateOperation<tim::vx::ops::FullyConnected>(
      2, fc3_weight_shape[1]);
  (*fc3)
      .BindInputs({pool2_output, fc3_weight, fc3_bias})
      .BindOutputs({fc3_output});

  auto relu = graph->CreateOperation<tim::vx::ops::Relu>();
  (*relu).BindInput(fc3_output).BindOutput(relu_output);

  auto fc4 = graph->CreateOperation<tim::vx::ops::FullyConnected>(
      0, fc4_weight_shape[1]);
  (*fc4)
      .BindInputs({relu_output, fc4_weight, fc4_bias})
      .BindOutputs({fc4_output});

  auto softmax = graph->CreateOperation<tim::vx::ops::Softmax>(1.0f, 0);
  (*softmax).BindInput(fc4_output).BindOutput(output);

  if (!graph->Compile()) {
    std::cout << "Compile graph fail." << std::endl;
    return -1;
  }

  if (!input->CopyDataToTensor(input_data.data(), input_data.size())) {
    std::cout << "Copy input data fail." << std::endl;
    return -1;
  }

  if (!graph->Run()) {
    std::cout << "Run graph fail." << std::endl;
    return -1;
  }

  std::vector<float> output_data;
  output_data.resize(1 * 10);
  if (!output->CopyDataFromTensor(output_data.data())) {
    std::cout << "Copy output data fail." << std::endl;
    return -1;
  }

  printTopN(output_data.data(), output_data.size(), 5);

  return 0;
}
