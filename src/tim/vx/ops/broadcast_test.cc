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
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/broadcast.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"
#include "test_utils.h"
#include "vsi_nn_pub.h"

#ifdef VSI_EXPAND_BROADCAST_ENABLE_DIMENSIONS
static void CheckResult(std::shared_ptr<tim::vx::Graph>& graph,
                 std::vector<float>& golden,
                 std::shared_ptr<tim::vx::Tensor>& output_tensor) {
  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size() * sizeof(float));
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Broadcast, ScalarTo2D_2x3) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1});
  tim::vx::ShapeType output_shape({3, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      2.25f,
  };
  std::vector<float> golden = {
      2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f,
  };
  std::vector<int32_t> shape = {3, 2};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 1DTo2D) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3});
  tim::vx::ShapeType output_shape({3, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 2.f, 3.f,
  };
  std::vector<float> golden = {
      1.f, 2.f, 3.f, 1.f, 2.f, 3.f,
  };
  std::vector<int32_t> shape = {3, 2};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 1DTo2D_WithDims0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2});
  tim::vx::ShapeType output_shape({2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 2.f,
  };
  std::vector<float> golden = {
      1.f, 2.f,
      1.f, 2.f,
  };
  std::vector<int32_t> shape = {2, 2};
  std::vector<int32_t> dimensions = {0};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 1DTo2D_WithDims1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2});
  tim::vx::ShapeType output_shape({2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 2.f,
  };
  std::vector<float> golden = {
      1.f, 1.f,
      2.f, 2.f,
  };
  std::vector<int32_t> shape = {2, 2};
  std::vector<int32_t> dimensions = {1};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 1DTo3D_WithDims0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2});
  tim::vx::ShapeType output_shape({2, 2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 2.f,
  };
  std::vector<float> golden = {
      1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f,
  };
  std::vector<int32_t> shape = {2, 2, 2};
  std::vector<int32_t> dimensions = {0};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 1DTo3D_WithDims1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2});
  tim::vx::ShapeType output_shape({2, 2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 2.f,
  };
  std::vector<float> golden = {
      1.f, 1.f, 2.f, 2.f, 1.f, 1.f, 2.f, 2.f,
  };
  std::vector<int32_t> shape = {2, 2, 2};
  std::vector<int32_t> dimensions = {1};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 1DTo3D_WithDims2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2});
  tim::vx::ShapeType output_shape({2, 2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 2.f,
  };
  std::vector<float> golden = {
      1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f,
  };
  std::vector<int32_t> shape = {2, 2, 2};
  std::vector<int32_t> dimensions = {2};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 2DTo3D_WithDims02) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2});
  tim::vx::ShapeType output_shape({2, 2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 5.f, 2.f, 6.f
  };
  std::vector<float> golden = {
      1.f, 5.f, 1.f, 5.f, 2.f, 6.f, 2.f, 6.f,
  };
  std::vector<int32_t> shape = {2, 2, 2};
  std::vector<int32_t> dimensions = {0, 2};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}

TEST(Broadcast, 2DTo3D_WithDims12) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2});
  tim::vx::ShapeType output_shape({2, 2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1.f, 5.f, 2.f, 6.f
  };
  std::vector<float> golden = {
      1.f, 1.f, 5.f, 5.f, 2.f, 2.f, 6.f, 6.f,
  };
  std::vector<int32_t> shape = {2, 2, 2};
  std::vector<int32_t> dimensions = {1, 2};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape, dimensions);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  CheckResult(graph, golden, output_tensor);
}
#endif
