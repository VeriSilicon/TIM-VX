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
#include <algorithm>
#include <stdarg.h>
#include "tim/transform/mean_stddev_normalize_fusion.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/layernormalization.h"
#include "tim/vx/ops/instancenormalization.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {
enum {
  NORMALIZATION_INDEX_MEAN_0 = 0,
  NORMALIZATION_INDEX_SUB_0 = 1,
  NORMALIZATION_INDEX_MUL_2 = 2,
  NORMALIZATION_INDEX_POW = 3,
  NORMALIZATION_INDEX_MEAN_1 = 4,
  NORMALIZATION_INDEX_ADD_0 = 5,
  NORMALIZATION_INDEX_RSQRT = 6,
  NORMALIZATION_INDEX_MUL_0 = 7,
  NORMALIZATION_INDEX_MUL_1 = 8,
  NORMALIZATION_INDEX_ADD_1 = 9,
  NORMALIZATION_INDEX_SUB_1 = 10
};

//Determine whether the needed opkind is in given consumer op list
bool OpkindInConsumers(std::vector<std::shared_ptr<vx::Operation>> consumers,
                       int32_t op_id) {
  if (consumers.size() == 1 && consumers[0]->impl()->kind_ != op_id) {
    return false;
  }
  auto op_iter = std::find_if(consumers.begin(), consumers.end(),
                              [op_id](std::shared_ptr<vx::Operation> oper) {
                                return oper.get()->impl()->kind_ == op_id;
                              });
  return op_iter != consumers.end();
}

// Check if one of op's consumers has already in list.
// Only if the consumer is same to the compared one with given index
// can be considered as pattern matched.
bool OpInConsumer(const std::shared_ptr<vx::Graph>& graph,
                  const std::shared_ptr<vx::Operation>& current,
                  const std::shared_ptr<vx::Operation>& compared) {
  auto output_tensor = current->impl()->OutputsTensor()[0];
  auto ops = graph->GetConsumersOp(output_tensor);
  for (auto op : ops) {
    if (op == compared) {
      return true;
    }
  }
  return false;
}

// Determine whether the current op is suitable for pattern matching with specified
// consumers. The possible ops will be stored in a temporary vector created during
// each Pattern Matching. Special situation that the consumer has already in list will
// NOT be concerned in this function.
bool UpdateTempVector(std::vector<std::shared_ptr<vx::Operation>>& temp,
                      int32_t curr_index,
                      const std::shared_ptr<vx::Graph>& graph,
                      std::vector<int32_t> op_kind) {
  auto outputs = temp[curr_index]->impl()->OutputsTensor();
  auto ops = graph->GetConsumersOp(outputs[0]);
  if (outputs.size() > 1 || ops.size() != op_kind.size() || op_kind.size() > 2)
    return false;
  else {
    for (int32_t op_k : op_kind) {
      if (!OpkindInConsumers(ops, op_k)) return false;
    }
    if (op_kind.size() == 2) {
      int32_t first_index = ops[0]->impl()->kind_ == op_kind[0] ? 0 : 1;
      //push back ops as same order as need
      temp.push_back(graph->GetConsumersOp(outputs[0])[first_index]);
      temp.push_back(graph->GetConsumersOp(outputs[0])[1 - first_index]);
    } else {
      temp.push_back(ops[0]);
    }
    return true;
  }
}

// Remove ops and tensors in each matched normlization patten
void RemoveTensorsAndOps(
    std::shared_ptr<vx::Graph>& graph,
    const std::vector<std::shared_ptr<vx::Operation>>& norm_ops) {
  for (uint32_t i = 0; i < norm_ops.size(); i++) {
    auto it =
        std::remove_if(graph->OpVector().begin(), graph->OpVector().end(),
                       [norm_ops, i](std::shared_ptr<vx::Operation> oper) {
                         return oper == norm_ops[i];
                       });
    graph->OpVector().erase(it);  //Remove current op from op_vector_
    auto input_tensors = norm_ops[i]->impl()->InputsTensor();
    auto output_tensors = norm_ops[i]->impl()->OutputsTensor();

    switch (i) {
      case NORMALIZATION_INDEX_MEAN_0:
      case NORMALIZATION_INDEX_SUB_0:
      case NORMALIZATION_INDEX_MUL_1:
        for (auto tensor : input_tensors) {
          if (tensor->GetSpec().attr_ == vx::TensorAttribute::CONSTANT)
            graph->TensorConsumer().erase(tensor);
          else {
            it = std::find_if(
                graph->TensorConsumer()[tensor].begin(),
                graph->TensorConsumer()[tensor].end(),
                [i, norm_ops](std::shared_ptr<vx::Operation> oper) {
                  return oper == norm_ops[i];
                });
            if (it != graph->TensorConsumer()[tensor].end())
              graph->TensorConsumer()[tensor].erase(it);
            if (graph->TensorConsumer()[tensor].empty())
              graph->TensorConsumer().erase(tensor);
          }
          graph->TensorProducer().erase(output_tensors[0]);
        }
        break;
      case NORMALIZATION_INDEX_ADD_1:
        break;
      default:
        for (auto tensor : input_tensors) {
          if (tensor->GetSpec().attr_ != vx::TensorAttribute::CONSTANT) {
            if (graph->TensorProducer()[tensor] != nullptr) {
              auto it =
                  std::find(graph->OpVector().begin(), graph->OpVector().end(),
                            graph->GetProducerOp(tensor));
              graph->OpVector().erase(it);
              graph->TensorProducer().erase(tensor);
            }
          }
          graph->TensorConsumer().erase(tensor);
          for (auto tensor : output_tensors)
            graph->TensorProducer().erase(tensor);
        }
        break;
    }
  }
}

bool CheckMul0(const std::shared_ptr<vx::Graph>& graph,
               std::vector<std::shared_ptr<vx::Operation>>& norm_ops) {
  auto mul0_output_tensor =
      norm_ops[NORMALIZATION_INDEX_MUL_0]->impl()->OutputsTensor();
  auto mul0_consumers = graph->GetConsumersOp(mul0_output_tensor[0]);
  if (mul0_output_tensor.size() > 1 || mul0_consumers.size() != 2 ||
      mul0_consumers[0]->impl()->kind_ != 1 ||
      mul0_consumers[1]->impl()->kind_ != 1)
    return false;
  if (!OpInConsumer(graph, norm_ops[NORMALIZATION_INDEX_MUL_0],
                    norm_ops[NORMALIZATION_INDEX_MUL_2]))
    return false;
  int32_t mul1_index = graph->GetConsumersOp(mul0_output_tensor[0])[0] ==
                               norm_ops[NORMALIZATION_INDEX_MUL_2]
                           ? 1
                           : 0;
  norm_ops.push_back(mul0_consumers[mul1_index]);
  return true;
}

bool CheckMean0Sub0Mul1SameInput(
    std::vector<std::shared_ptr<vx::Operation>>& norm_ops) {
  auto mean0_inputs_tensors =
      norm_ops[NORMALIZATION_INDEX_MEAN_0]->impl()->InputsTensor();
  auto sub0_inputs_tensors =
      norm_ops[NORMALIZATION_INDEX_SUB_0]->impl()->InputsTensor();
  auto mul1_inputs_tensors =
      norm_ops[NORMALIZATION_INDEX_MUL_1]->impl()->InputsTensor();
  std::sort(mean0_inputs_tensors.begin(), mean0_inputs_tensors.end());
  std::sort(sub0_inputs_tensors.begin(), sub0_inputs_tensors.end());
  std::sort(mul1_inputs_tensors.begin(), mul1_inputs_tensors.end());
  std::vector<std::shared_ptr<vx::Tensor>> intersect1, intersect2;
  std::set_intersection(mean0_inputs_tensors.begin(),
                        mean0_inputs_tensors.end(), sub0_inputs_tensors.begin(),
                        sub0_inputs_tensors.end(),
                        std::back_inserter(intersect1));
  std::set_intersection(sub0_inputs_tensors.begin(), sub0_inputs_tensors.end(),
                        mul1_inputs_tensors.begin(), mul1_inputs_tensors.end(),
                        std::back_inserter(intersect2));
  if (intersect2.empty())
    return false;
  else
    return true;
}

void LayernormConnection(std::shared_ptr<vx::Graph>& graph,
                         std::vector<std::shared_ptr<vx::Operation>> norm_ops) {
  auto src_tensor =
      norm_ops[NORMALIZATION_INDEX_MEAN_0]->impl()->InputsTensor()[0];
  auto final_tensor =
      norm_ops[NORMALIZATION_INDEX_ADD_1]->impl()->OutputsTensor()[0];
  int32_t axis = *norm_ops[NORMALIZATION_INDEX_MEAN_0]
                      ->impl()
                      ->node()
                      ->nn_param.reduce.axis;
  axis = src_tensor->GetShape().size() - axis - 1;  // reverse axis

  // Get eps, gamma,beta;
  // Do datatype convert due to Layernorm op requirements
  int32_t eps_index =
      graph->GetProducerOp(
          norm_ops[NORMALIZATION_INDEX_ADD_0]->impl()->InputsTensor()[0]) ==
              norm_ops[NORMALIZATION_INDEX_MEAN_1]
          ? 1
          : 0;
  auto org_eps =
      norm_ops[NORMALIZATION_INDEX_ADD_0]->impl()->InputsTensor()[eps_index];
  if (!org_eps->IsConstTensor()) {
    org_eps = graph->GetProducerOp(org_eps)->impl()->InputsTensor()[0];
  }
  auto org_gamma =
      norm_ops[NORMALIZATION_INDEX_MUL_0]->impl()->InputsTensor()[1];
  auto org_beta =
      norm_ops[NORMALIZATION_INDEX_SUB_1]->impl()->InputsTensor()[0];

  float* float_eps = org_eps->ConvertTensorToFloat32Data();
  float* float_gamma = org_gamma->ConvertTensorToFloat32Data();
  float* float_beta = org_beta->ConvertTensorToFloat32Data();
  RemoveTensorsAndOps(graph, norm_ops);
  std::vector<uint32_t> shape(src_tensor->GetShape().size(), 1);
  shape[axis] = src_tensor->GetShape()[axis];
  vx::TensorSpec param_spec(vx::DataType::FLOAT32, shape,
                            vx::TensorAttribute::CONSTANT);

  auto beta = graph->CreateTensor(param_spec, float_beta);
  auto gamma = graph->CreateTensor(param_spec, float_gamma);
  float eps = *float_eps;
  beta->CopyDataToTensor(float_beta);
  gamma->CopyDataToTensor(float_gamma);
  vsi_nn_Free(float_gamma);
  vsi_nn_Free(float_beta);
  vsi_nn_Free(float_eps);

  auto layernorm =
      graph->CreateOperation<vx::ops::LayerNormalization>(axis, eps);

  graph->TensorConsumer()[src_tensor].push_back(layernorm);
  layernorm->BindInputs({src_tensor, beta, gamma});
  layernorm->BindOutputs({final_tensor});
}

void InstancenormConnection(
    std::shared_ptr<vx::Graph>& graph,
    const std::vector<std::shared_ptr<vx::Operation>>& norm_ops) {
  auto src_tensor =
      norm_ops[NORMALIZATION_INDEX_MEAN_0]->impl()->InputsTensor()[0];
  auto final_tensor =
      norm_ops[NORMALIZATION_INDEX_ADD_1]->impl()->OutputsTensor()[0];

  // Get eps, gamma,beta from graph.
  // Do datatype convert due to InstanceNormlization op requirements
  int32_t eps_index =
      graph->GetProducerOp(
          norm_ops[NORMALIZATION_INDEX_ADD_0]->impl()->InputsTensor()[0]) ==
              norm_ops[NORMALIZATION_INDEX_MEAN_1]
          ? 1
          : 0;
  auto org_eps =
      norm_ops[NORMALIZATION_INDEX_ADD_0]->impl()->InputsTensor()[eps_index];
  if (!org_eps->IsConstTensor()) {
    org_eps = graph->GetProducerOp(org_eps)->impl()->InputsTensor()[0];
  }

  auto org_gamma =
      norm_ops[NORMALIZATION_INDEX_MUL_0]->impl()->InputsTensor()[1];
  auto org_beta =
      norm_ops[NORMALIZATION_INDEX_SUB_1]->impl()->InputsTensor()[0];

  float* float_eps = org_eps->ConvertTensorToFloat32Data();
  float* float_gamma = org_gamma->ConvertTensorToFloat32Data();
  float* float_beta = org_beta->ConvertTensorToFloat32Data();
  RemoveTensorsAndOps(graph, norm_ops);

  std::vector<uint32_t> shape(src_tensor->GetShape().size(), 1);
  shape[0] = src_tensor->GetShape()[0];
  vx::TensorSpec param_spec(vx::DataType::FLOAT32, shape,
                            vx::TensorAttribute::CONSTANT);

  auto beta = graph->CreateTensor(param_spec, float_beta);
  auto gamma = graph->CreateTensor(param_spec, float_gamma);
  float eps = *float_eps;
  vsi_nn_Free(float_gamma);
  vsi_nn_Free(float_beta);
  vsi_nn_Free(float_eps);

  auto instancenorm = graph->CreateOperation<vx::ops::InstanceNormalization>(
      eps, vx::DataLayout::CWHN);
  graph->TensorConsumer()[src_tensor].push_back(instancenorm);
  instancenorm->BindInputs({src_tensor, beta, gamma});
  instancenorm->BindOutputs({final_tensor});
}

/* Checking Mean StdDev Normalization structure:
         input
        /  |  \
       /   |   Mean0
      |    |   / |
      |   Sub0   |
      |    |     |
      |   Pow    |
      |    |     |
      |   Mean1  |
      |    |     |
      |   Add0   |
      |    |     |
      |   Rsqrt  |
      |    |     |
      |   Mul0   |
      |  /    \  |
     Mul1      Mul2
      |         |
      |       Sub1
       \      /
         Add1
          |
        output
*/
void MeanStdDevNormalization(std::shared_ptr<vx::Graph>& graph) {
  std::vector<std::shared_ptr<vx::Operation>>& op_vector = graph->OpVector();

  for (auto& op : op_vector) {
    if (std::find(op_vector.begin(), op_vector.end(), op) == op_vector.end())
      continue;  //Avoid read deleted data in op_vector
    if (op->impl()->kind_ != VSI_NN_OP_REDUCE) continue;

    std::vector<std::shared_ptr<vx::Operation>> temp;
    temp.push_back(op);

    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_MEAN_0, graph,
                          {VSI_NN_OP_SUBTRACT, VSI_NN_OP_MULTIPLY}))
      continue;

    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_SUB_0, graph,
                          {VSI_NN_OP_POW}))
      continue;

    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_POW, graph,
                          {VSI_NN_OP_REDUCE}))
      continue;  //Mean1
    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_MEAN_1, graph,
                          {VSI_NN_OP_ADD}))
      continue;  //Add0
    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_ADD_0, graph,
                          {VSI_NN_OP_RSQRT}))
      continue;  //Rsqrt
    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_RSQRT, graph,
                          {VSI_NN_OP_MULTIPLY}))
      continue;  //Mul0

    if (!CheckMul0(graph, temp)) continue;
    if (!CheckMean0Sub0Mul1SameInput(temp)) continue;

    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_MUL_1, graph,
                          {VSI_NN_OP_ADD}))
      continue;  //Add1
    if (!UpdateTempVector(temp, NORMALIZATION_INDEX_MUL_2, graph,
                          {VSI_NN_OP_SUBTRACT}))  //Sub1
      continue;

    auto sub_outputs = temp[NORMALIZATION_INDEX_SUB_1]->impl()->OutputsTensor();
    if (sub_outputs.size() >= 2 ||
        graph->GetConsumersOp(sub_outputs[0]).size() > 1 ||
        graph->GetConsumersOp(sub_outputs[0])[0]->impl()->kind_ != 0)
      continue;
    if (!OpInConsumer(graph, temp[NORMALIZATION_INDEX_SUB_1],
                      temp[NORMALIZATION_INDEX_ADD_1]))
      continue;

    int axis_num = temp[NORMALIZATION_INDEX_MEAN_0]
                       ->impl()
                       ->node()
                       ->nn_param.reduce.axis_num;
    if (axis_num == 1) {
      LayernormConnection(graph, temp);
    } else {
      InstancenormConnection(graph, temp);
    }
  }
}
}  // namespace transform
}  // namespace tim
