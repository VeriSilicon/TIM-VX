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
#include "tim/vx/ops/bidirectional_sequence_lstm.h"

#include "gtest/gtest.h"
#include "test_utils.h"

std::shared_ptr<tim::vx::Tensor> make_empty_tensor(
    std::shared_ptr<tim::vx::Graph> graph, const tim::vx::ShapeType& shape,
    const tim::vx::TensorAttribute& role);  //, const float& default_value)

TEST(Bidirectional_LSTM_CELL, shape_in_2_cell_4_out_4_float32) {
    // NoCifg_NoPeephole_NoProjection_NoLayerNorm
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    uint32_t n_batch, n_step, n_cell, n_input, n_output;
    n_batch = 1, n_step = 3, n_cell = 4, n_input = 2, n_output = 4;
    tim::vx::ShapeType input_shape, cell_shape, state_shape;
    input_shape = {n_batch, n_step, n_input}; // non-time-major

    tim::vx::TensorSpec lstm_input_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_step, n_batch}), tim::vx::TensorAttribute::INPUT);

    tim::vx::TensorSpec fw_weight_i2i_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_weight_i2f_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_weight_i2c_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_weight_i2o_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);

    tim::vx::TensorSpec fw_weight_r2i_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_weight_r2f_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_weight_r2c_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_weight_r2o_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);

    tim::vx::TensorSpec fw_bias_i_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_bias_f_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_bias_c_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec fw_bias_o_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);

    tim::vx::TensorSpec bw_weight_i2i_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_weight_i2f_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_weight_i2c_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_weight_i2o_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_cell}), tim::vx::TensorAttribute::CONSTANT);

    tim::vx::TensorSpec bw_weight_r2i_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_weight_r2f_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_weight_r2c_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_weight_r2o_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_cell}), tim::vx::TensorAttribute::CONSTANT);

    tim::vx::TensorSpec bw_bias_i_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_bias_f_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_bias_c_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bw_bias_o_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell}), tim::vx::TensorAttribute::CONSTANT);

    tim::vx::TensorSpec fw_output_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_step, n_batch}), tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec bw_output_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_output, n_step, n_batch}), tim::vx::TensorAttribute::OUTPUT);

    auto lstm_input = graph->CreateTensor(lstm_input_spec);
    std::vector<float> lstm_input_data = {2., 3., 3., 4., 1., 1.};
    lstm_input->CopyDataToTensor(lstm_input_data.data(), lstm_input_data.size() * 4);

    auto fw_output_tensor = graph->CreateTensor(fw_output_spec);
    auto bw_output_tensor = graph->CreateTensor(bw_output_spec);

    std::vector<float> fw_weight_i2i = {-0.45018822, -0.02338299, -0.0870589,
                                        -0.34550029, 0.04266912, -0.15680569,
                                        -0.34856534, 0.43890524};
    std::vector<float> fw_weight_i2f = {0.09701663, 0.20334584, -0.50592935,
                                        -0.31343272, -0.40032279, 0.44781327,
                                        0.01387155, -0.35593212};
    std::vector<float> fw_weight_i2c = {-0.50013041, 0.1370284, 0.11810488, 0.2013163,
                                        -0.20583314, 0.44344562, 0.22077113,
                                        -0.29909778};
    std::vector<float> fw_weight_i2o = {-0.25065863, -0.28290087, 0.04613829,
                                        0.40525138, 0.44272184, 0.03897077, -0.1556896,
                                        0.19487578};
    auto fw_weight_i2i_tensor = graph->CreateTensor(fw_weight_i2i_spec, fw_weight_i2i.data());
    auto fw_weight_i2f_tensor = graph->CreateTensor(fw_weight_i2f_spec, fw_weight_i2f.data());
    auto fw_weight_i2c_tensor = graph->CreateTensor(fw_weight_i2c_spec, fw_weight_i2c.data());
    auto fw_weight_i2o_tensor = graph->CreateTensor(fw_weight_i2o_spec, fw_weight_i2o.data());

    std::vector<float> fw_weight_r2i = {
        -0.0063535,  -0.2042388,  0.31454784,  -0.35746509,
        0.28902304,  0.08183324,  -0.16555229, 0.02286911,
        -0.13566875, 0.03034258,  0.48091322,  -0.12528998,
        0.24077177,  -0.51332325, -0.33502164, 0.10629296};
    std::vector<float> fw_weight_r2f = {
        -0.48684245, -0.06655136, 0.42224967,  0.2112639,
        0.27654213,  0.20864892,  -0.07646349, 0.45877004,
        0.00141793,  -0.14609534, 0.36447752,  0.09196436,
        0.28053468,  0.01560611,  -0.20127171, -0.01140004};
    std::vector<float> fw_weight_r2c = {
        -0.3407414,  0.24443203,  -0.2078532,  0.26320225,
        0.05695659,  -0.00123841, -0.4744786,  -0.35869038,
        -0.06418842, -0.13502428, -0.501764,   0.22830659,
        -0.46367589, 0.26016325,  -0.03894562, -0.16368064};
    std::vector<float> fw_weight_r2o = {
        0.43385774,  -0.17194885, 0.2718237,  0.09215671,
        0.24107647,  -0.39835793, 0.18212086, 0.01301402,
        0.48572797,  -0.50656658, 0.20047462, -0.20607421,
        -0.51818722, -0.15390486, 0.0468148,  0.39922136};

    auto fw_weight_r2i_tensor = graph->CreateTensor(fw_weight_r2i_spec, fw_weight_r2i.data());
    auto fw_weight_r2f_tensor = graph->CreateTensor(fw_weight_r2f_spec, fw_weight_r2f.data());
    auto fw_weight_r2c_tensor = graph->CreateTensor(fw_weight_r2c_spec, fw_weight_r2c.data());
    auto fw_weight_r2o_tensor = graph->CreateTensor(fw_weight_r2o_spec, fw_weight_r2o.data());

    std::vector<float> fw_bias_i = {0.0, 0.0, 0.0, 0.0};
    std::vector<float> fw_bias_f = {1., 1., 1., 1.};
    std::vector<float> fw_bias_c = {0.0, 0.0, 0.0, 0.0};
    std::vector<float> fw_bias_o = {0.0, 0.0, 0.0, 0.0};
    auto fw_bias_i_tensor = graph->CreateTensor(fw_bias_i_spec, fw_bias_i.data());
    auto fw_bias_f_tensor = graph->CreateTensor(fw_bias_f_spec, fw_bias_f.data());
    auto fw_bias_c_tensor = graph->CreateTensor(fw_bias_c_spec, fw_bias_c.data());
    auto fw_bias_o_tensor = graph->CreateTensor(fw_bias_o_spec, fw_bias_o.data());

    std::vector<float> bw_weight_i2i = {-0.45018822, -0.02338299, -0.0870589,
                                        -0.34550029, 0.04266912, -0.15680569,
                                        -0.34856534, 0.43890524};
    std::vector<float> bw_weight_i2f = {0.09701663, 0.20334584, -0.50592935,
                                        -0.31343272, -0.40032279, 0.44781327,
                                        0.01387155, -0.35593212};
    std::vector<float> bw_weight_i2c = {-0.50013041, 0.1370284, 0.11810488, 0.2013163,
                                        -0.20583314, 0.44344562, 0.22077113,
                                        -0.29909778};
    std::vector<float> bw_weight_i2o = {-0.25065863, -0.28290087, 0.04613829,
                                        0.40525138, 0.44272184, 0.03897077, -0.1556896,
                                        0.19487578};
    auto bw_weight_i2i_tensor = graph->CreateTensor(bw_weight_i2i_spec, bw_weight_i2i.data());
    auto bw_weight_i2f_tensor = graph->CreateTensor(bw_weight_i2f_spec, bw_weight_i2f.data());
    auto bw_weight_i2c_tensor = graph->CreateTensor(bw_weight_i2c_spec, bw_weight_i2c.data());
    auto bw_weight_i2o_tensor = graph->CreateTensor(bw_weight_i2o_spec, bw_weight_i2o.data());

    std::vector<float> bw_weight_r2i = {
        -0.0063535,  -0.2042388,  0.31454784,  -0.35746509,
        0.28902304,  0.08183324,  -0.16555229, 0.02286911,
        -0.13566875, 0.03034258,  0.48091322,  -0.12528998,
        0.24077177,  -0.51332325, -0.33502164, 0.10629296};
    std::vector<float> bw_weight_r2f = {
        -0.48684245, -0.06655136, 0.42224967,  0.2112639,
        0.27654213,  0.20864892,  -0.07646349, 0.45877004,
        0.00141793,  -0.14609534, 0.36447752,  0.09196436,
        0.28053468,  0.01560611,  -0.20127171, -0.01140004};
    std::vector<float> bw_weight_r2c = {
        -0.3407414,  0.24443203,  -0.2078532,  0.26320225,
        0.05695659,  -0.00123841, -0.4744786,  -0.35869038,
        -0.06418842, -0.13502428, -0.501764,   0.22830659,
        -0.46367589, 0.26016325,  -0.03894562, -0.16368064};
    std::vector<float> bw_weight_r2o = {
        0.43385774,  -0.17194885, 0.2718237,  0.09215671,
        0.24107647,  -0.39835793, 0.18212086, 0.01301402,
        0.48572797,  -0.50656658, 0.20047462, -0.20607421,
        -0.51818722, -0.15390486, 0.0468148,  0.39922136};

    auto bw_weight_r2i_tensor = graph->CreateTensor(bw_weight_r2i_spec, bw_weight_r2i.data());
    auto bw_weight_r2f_tensor = graph->CreateTensor(bw_weight_r2f_spec, bw_weight_r2f.data());
    auto bw_weight_r2c_tensor = graph->CreateTensor(bw_weight_r2c_spec, bw_weight_r2c.data());
    auto bw_weight_r2o_tensor = graph->CreateTensor(bw_weight_r2o_spec, bw_weight_r2o.data());

    std::vector<float> bw_bias_i = {0.0, 0.0, 0.0, 0.0};
    std::vector<float> bw_bias_f = {1., 1., 1., 1.};
    std::vector<float> bw_bias_c = {0.0, 0.0, 0.0, 0.0};
    std::vector<float> bw_bias_o = {0.0, 0.0, 0.0, 0.0};
    auto bw_bias_i_tensor = graph->CreateTensor(bw_bias_i_spec, bw_bias_i.data());
    auto bw_bias_f_tensor = graph->CreateTensor(bw_bias_f_spec, bw_bias_f.data());
    auto bw_bias_c_tensor = graph->CreateTensor(bw_bias_c_spec, bw_bias_c.data());
    auto bw_bias_o_tensor = graph->CreateTensor(bw_bias_o_spec, fw_bias_o.data());

    auto bidirectional_lstm = graph->CreateOperation<tim::vx::ops::BidirectionalSequenceLstm>(
        0.0, 0.0, tim::vx::ops::BidirectionalSequenceLstm::ActivationType::kTANH, 0.0, false,
        tim::vx::ops::BidirectionalSequenceLstm::kSIGMOID, true);

    (*bidirectional_lstm)
        .BindInputs({
            lstm_input,

            fw_weight_i2i_tensor,
            fw_weight_i2f_tensor,
            fw_weight_i2c_tensor,
            fw_weight_i2o_tensor,

            fw_weight_r2i_tensor,
            fw_weight_r2f_tensor,
            fw_weight_r2c_tensor,
            fw_weight_r2o_tensor,

            graph->CreateTensorPlaceHolder(),       /*fw_weight_c2i*/
            graph->CreateTensorPlaceHolder(),       /*fw_weight_c2f*/
            graph->CreateTensorPlaceHolder(),       /*fw_weight_c2o*/

            fw_bias_i_tensor,
            fw_bias_f_tensor,
            fw_bias_c_tensor,
            fw_bias_o_tensor,

            // optional for projection
            graph->CreateTensorPlaceHolder(),       /*fw_weight_prj*/
            graph->CreateTensorPlaceHolder(),       /*fw_bias_prj*/

            bw_weight_i2i_tensor,
            bw_weight_i2f_tensor,
            bw_weight_i2c_tensor,
            bw_weight_i2o_tensor,

            bw_weight_r2i_tensor,
            bw_weight_r2f_tensor,
            bw_weight_r2c_tensor,
            bw_weight_r2o_tensor,

            graph->CreateTensorPlaceHolder(),       /*bw_weight_c2i*/
            graph->CreateTensorPlaceHolder(),       /*bw_weight_c2f*/
            graph->CreateTensorPlaceHolder(),       /*bw_weight_c2o*/

            bw_bias_i_tensor,
            bw_bias_f_tensor,
            bw_bias_c_tensor,
            bw_bias_o_tensor,

            // optional for projection
            graph->CreateTensorPlaceHolder(),       /*bw_weight_prj*/
            graph->CreateTensorPlaceHolder(),       /*bw_bias_prj*/

            graph->CreateTensorPlaceHolder(),       /*fw_h_state*/
            graph->CreateTensorPlaceHolder(),       /*fw_c_state*/
            graph->CreateTensorPlaceHolder(),       /*bw_h_state*/
            graph->CreateTensorPlaceHolder(),       /*bw_c_state*/

            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),       // AUX

            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),       // Layer_norm
        })
        .BindOutputs({
            fw_output_tensor,
            make_empty_tensor(
                graph, tim::vx::ShapeType({n_output, n_batch}), tim::vx::TensorAttribute::OUTPUT),  /*fw_h_state*/
            make_empty_tensor(
                graph, tim::vx::ShapeType({n_cell, n_batch}), tim::vx::TensorAttribute::OUTPUT),  /*fw_c_state*/

            bw_output_tensor,
            make_empty_tensor(
                graph, tim::vx::ShapeType({n_output, n_batch}), tim::vx::TensorAttribute::OUTPUT),  /*bw_h_state*/
            make_empty_tensor(
                graph, tim::vx::ShapeType({n_cell, n_batch}), tim::vx::TensorAttribute::OUTPUT),  /*bw_c_state*/
        });

    graph->Compile();
    graph->Run();

    std::vector<float> lstm_fw_golden_output = {
      -0.02973187, 0.1229473,  0.20885126, -0.15358765,
      -0.03716109, 0.12507336, 0.41193449, -0.20860538,
      -0.15053082, 0.09120187, 0.24278517, -0.12222792};
    std::vector<float> lstm_bw_golden_output = {
      -0.0806187, 0.139077, 0.400476,   -0.197842, 
      -0.0332076, 0.123838, 0.309777,   -0.17621, 
      -0.0490733, 0.0739237, 0.067706,   -0.0208124};
    std::vector<float> fw_output(lstm_fw_golden_output.size());
    std::vector<float> bw_output(lstm_bw_golden_output.size());
    fw_output_tensor->CopyDataFromTensor(fw_output.data());
    bw_output_tensor->CopyDataFromTensor(bw_output.data());

    EXPECT_TRUE(ArraysMatch(lstm_fw_golden_output, fw_output, 1e-3f));
    EXPECT_TRUE(ArraysMatch(lstm_bw_golden_output, bw_output, 1e-3f));
}