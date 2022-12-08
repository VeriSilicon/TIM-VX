/****************************************************************************
*
*    Copyright (c) 2022 Vivante Corporation
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
#include "tim/vx/ops.h"

#include "gtest/gtest.h"
#include "test_utils.h"

std::shared_ptr<tim::vx::Tensor> make_empty_tensor(
    std::shared_ptr<tim::vx::Graph> graph, const tim::vx::ShapeType& shape,
    const tim::vx::TensorAttribute& role);  //, const float& default_value)

TEST(Bidirectional_LSTM_CELL_EXT, shape_in_2_cell_4_out_4_float32) {
    // NoCifg_NoPeephole_NoProjection_NoLayerNorm
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    uint32_t n_batch, n_step, n_cell, n_input, n_directions;
    n_batch = 1, n_step = 3, n_cell = 4, n_input = 2, n_directions=2;
    tim::vx::ShapeType input_shape, cell_shape, state_shape;
    input_shape = {n_batch, n_step, n_input}; // no-time-major

    tim::vx::TensorSpec lstm_input_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, n_step, n_batch}), tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec i_weights_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_input, 4*n_cell, n_directions}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec r_weights_spec(tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell, 4*n_cell, n_directions}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bias_spec  (tim::vx::DataType::FLOAT32, tim::vx::ShapeType({8*n_cell, n_directions}), tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec output_spec    (tim::vx::DataType::FLOAT32, tim::vx::ShapeType({n_cell, n_directions, n_step, n_batch}), tim::vx::TensorAttribute::OUTPUT);


    auto lstm_input = graph->CreateTensor(lstm_input_spec);
    std::vector<float> lstm_input_data = {2., 3., 3., 4., 1., 1.};
    lstm_input->CopyDataToTensor(lstm_input_data.data(), lstm_input_data.size() * 4);

    auto output_tensor = graph->CreateTensor(output_spec);


    std::vector<float> input_weights = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                      0.04266912,  -0.15680569, -0.34856534, 0.43890524,
                      -0.25065863, -0.28290087, 0.04613829, 0.40525138,
                      0.44272184,  0.03897077,  -0.1556896, 0.19487578,
                      0.09701663,  0.20334584,  -0.50592935, -0.31343272,
                      -0.40032279, 0.44781327, 0.01387155,  -0.35593212,
                      -0.50013041, 0.1370284,  0.11810488, 0.2013163,
                      -0.20583314, 0.44344562, 0.22077113, -0.29909778,
                      -0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                      0.04266912,  -0.15680569, -0.34856534, 0.43890524,
                      -0.25065863, -0.28290087, 0.04613829, 0.40525138,
                      0.44272184,  0.03897077,  -0.1556896, 0.19487578,
                      0.09701663,  0.20334584,  -0.50592935, -0.31343272,
                      -0.40032279, 0.44781327, 0.01387155,  -0.35593212,
                      -0.50013041, 0.1370284,  0.11810488, 0.2013163,
                      -0.20583314, 0.44344562, 0.22077113, -0.29909778};
    auto input_weights_tensor = graph->CreateTensor(i_weights_spec, input_weights.data());
    


    std::vector<float> recurrent_weights = {
        -0.0063535,  -0.2042388,  0.31454784,  -0.35746509,
        0.28902304,  0.08183324,  -0.16555229, 0.02286911,
        -0.13566875, 0.03034258,  0.48091322,  -0.12528998,
        0.24077177,  -0.51332325, -0.33502164, 0.10629296,
        0.43385774,  -0.17194885, 0.2718237,  0.09215671,
        0.24107647,  -0.39835793, 0.18212086, 0.01301402,
        0.48572797,  -0.50656658, 0.20047462, -0.20607421,
        -0.51818722, -0.15390486, 0.0468148,  0.39922136,
        -0.48684245, -0.06655136, 0.42224967,  0.2112639,
        0.27654213,  0.20864892,  -0.07646349, 0.45877004,
        0.00141793,  -0.14609534, 0.36447752,  0.09196436,
        0.28053468,  0.01560611,  -0.20127171, -0.01140004,
        -0.3407414,  0.24443203,  -0.2078532,  0.26320225,
        0.05695659,  -0.00123841, -0.4744786,  -0.35869038,
        -0.06418842, -0.13502428, -0.501764,   0.22830659,
        -0.46367589, 0.26016325,  -0.03894562, -0.16368064,
        -0.0063535,  -0.2042388,  0.31454784,  -0.35746509,
        0.28902304,  0.08183324,  -0.16555229, 0.02286911,
        -0.13566875, 0.03034258,  0.48091322,  -0.12528998,
        0.24077177,  -0.51332325, -0.33502164, 0.10629296,
        0.43385774,  -0.17194885, 0.2718237,  0.09215671,
        0.24107647,  -0.39835793, 0.18212086, 0.01301402,
        0.48572797,  -0.50656658, 0.20047462, -0.20607421,
        -0.51818722, -0.15390486, 0.0468148,  0.39922136,
        -0.48684245, -0.06655136, 0.42224967,  0.2112639,
        0.27654213,  0.20864892,  -0.07646349, 0.45877004,
        0.00141793,  -0.14609534, 0.36447752,  0.09196436,
        0.28053468,  0.01560611,  -0.20127171, -0.01140004,
        -0.3407414,  0.24443203,  -0.2078532,  0.26320225,
        0.05695659,  -0.00123841, -0.4744786,  -0.35869038,
        -0.06418842, -0.13502428, -0.501764,   0.22830659,
        -0.46367589, 0.26016325,  -0.03894562, -0.16368064
        };
    auto r_weights_tensor = graph->CreateTensor(r_weights_spec, recurrent_weights.data());

    std::vector<float> bias = {0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               1., 1., 1., 1.,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               1., 1., 1., 1.,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0};
    auto bias_tensor = graph->CreateTensor(bias_spec, bias.data());


    auto bidirectional_lstm = graph->CreateOperation<tim::vx::ops::BidirectionalSequenceLstmExt>(
        0.0, tim::vx::ops::BidirectionalSequenceLstm::ActivationType::kTANH,
        false,
        tim::vx::ops::BidirectionalSequenceLstm::kSIGMOID);

    (*bidirectional_lstm)
        .BindInputs({
            lstm_input,
            input_weights_tensor,
            r_weights_tensor,
            bias_tensor,
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder(),
            graph->CreateTensorPlaceHolder()
        })
        .BindOutputs({
            output_tensor,
            make_empty_tensor(
                graph, tim::vx::ShapeType({n_cell, n_batch, n_directions}),
                tim::vx::TensorAttribute::OUTPUT),  //  Output_H_STATE
            make_empty_tensor(
                graph, tim::vx::ShapeType({n_cell, n_batch, n_directions}),
                tim::vx::TensorAttribute::OUTPUT),  //  output_C_State
        });
    graph->Compile();
    graph->Run();

    std::vector<float> lstm_golden_output = {
      -0.02973187, 0.1229473,  0.20885126, -0.15358765,
      -0.0806187, 0.139077, 0.400476,   -0.197842,
      -0.03716109, 0.12507336, 0.41193449, -0.20860538,
      -0.0332076, 0.123838, 0.309777,   -0.17621,
      -0.15053082, 0.09120187, 0.24278517, -0.12222792,
      -0.0490733, 0.0739237, 0.067706,   -0.0208124};
    
    std::vector<float> output(lstm_golden_output.size());
    output_tensor->CopyDataFromTensor(output.data());
   

    EXPECT_TRUE(ArraysMatch(lstm_golden_output, output, 1e-3f));
}