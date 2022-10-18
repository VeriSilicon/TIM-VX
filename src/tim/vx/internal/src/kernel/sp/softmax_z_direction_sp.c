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

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_sp_unit_operation.h"
#include "kernel/vsi_nn_sp_lut.h"

#if (VX_STREAM_PROCESSOR_SUPPORT)

vsi_nn_spinst_t * vsi_nn_sp_max_axis2_inst
    (
        vx_context                context,
        int32_t                   fifo_depth,
        int32_t                   max_vector_depth
    )
{
    vsi_status status = VSI_FAILURE;
    const int32_t spInitInstsNum = 4;
    const int32_t spLoopInstsNum = fifo_depth > 4 ? 3 : 11;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;
    uint32_t f32_min = 0xff800000;
    float clampMin = *(float*)&f32_min;
    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[15];
    vsi_nn_spinst_attr_t attr;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* init inst0: r2 = -INF */
    status = vsi_nn_sp_move_constant(&sp_insts_param[0], clampMin, VSI_NN_SP_SR2);
    /* init inst1: r10 = 0 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 0, VSI_NN_SP_SR10);
    /* init inst2: r4 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[2], 1, VSI_NN_SP_SR4);
    /* init inst3: nop */
    status |= vsi_nn_sp_nop(&sp_insts_param[3]);
    CHECK_STATUS_FAIL_GOTO(status, final);

    if (fifo_depth > 4)
    {
        /* loop inst0: r1 = clamp(r3 * in, r6, r7) | r2 = v11 + r10 | r9 = r2 */
        status  = vsi_nn_sp_mul_clamp(&sp_insts_param[4], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_VR11, VSI_NN_SP_SR10, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[4], VSI_NN_SP_SR2, VSI_NN_SP_SR9);
        /* loop inst1: r8 = r1 * r4 | r5 = r1 - r2 | v11 = r5 ? r8 : r9 */
        status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR3, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_sub(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_move_sel0(&sp_insts_param[5], VSI_NN_SP_SR5, VSI_NN_SP_SR8, VSI_NN_SP_VR11);
        /* loop inst2: out = r1 */
        status |= vsi_nn_sp_move(&sp_insts_param[6], VSI_NN_SP_SR1, VSI_NN_SP_SROUT);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 7;

        attr.ignored_leading_outputs = 1;
        attr.ignored_leading_v11_rd = fifo_depth;
        attr.ignored_leading_v11_wr = 2;

        attr.num_of_v11_rd_in_flush_cycle = 0;
        attr.num_of_v11_wr_in_flush_cycle = 3;
    }
    else
    {
        /* loop inst0: r1 = clamp(r3 * in, r6, r7) | r2 = v11 + r10 | r9 = r2 */
        status  = vsi_nn_sp_mul_clamp(&sp_insts_param[4], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_VR11, VSI_NN_SP_SR10, VSI_NN_SP_SR2);
        /* loop inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[5]);
        /* loop inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[6]);
        /* loop inst3: r8 = r1 * r4 | r5 = r1 - r2 | v11 = r5 ? r8 : r9 */
        status |= vsi_nn_sp_mul(&sp_insts_param[7], VSI_NN_SP_SR1, VSI_NN_SP_SR3, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_sub(&sp_insts_param[7], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_move(&sp_insts_param[7], VSI_NN_SP_SR2, VSI_NN_SP_SR9);
        /* loop inst4: out = r1 */
        status |= vsi_nn_sp_move(&sp_insts_param[8], VSI_NN_SP_SR1, VSI_NN_SP_SROUT);
        /* loop inst5: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[9]);
        /* loop inst6: nop */
        status |= vsi_nn_sp_move_sel0(&sp_insts_param[10], VSI_NN_SP_SR5, VSI_NN_SP_SR8, VSI_NN_SP_VR11);
        /* loop inst7: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[11]);
        /* loop inst8: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[12]);
        /* loop inst9: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[13]);
        /* loop inst10: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[14]);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_outputs = 0;
        attr.ignored_leading_v11_rd = fifo_depth;
        attr.ignored_leading_v11_wr = 0;

        attr.num_of_v11_rd_in_flush_cycle = 0;
        attr.num_of_v11_wr_in_flush_cycle = 1;

        attr.flush_cycle_num = 10;
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst_by_context(context);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    return spinst;
}

DEF_SP_KERNEL_QUERY(max_axis2_query)
    (
    vsi_nn_kernel_node_t        node
    )
{
    vsi_status status = VSI_FAILURE;
    vx_size index = 0;
    vx_size tile_size[2] = {0};
    vsi_nn_spinst_t *spinst = NULL;
    int32_t fifo_depth = 0;
    int32_t max_vector_depth = 0;
    vx_context  ctx = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_ext2_t hw_param;

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_ext2_t));
    status = vxQueryHardwareCaps(ctx, (vx_hardware_caps_params_t*)(&hw_param), sizeof(vx_hardware_caps_params_ext2_t));
    CHECK_STATUS_FAIL_GOTO( status, final );

    status = vxQueryNode(node, VX_NODE_SWTILING_TILE_XY, tile_size, sizeof(tile_size));
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vxQueryNode(node, VX_NODE_SPINST_INDEX, &index, sizeof(index));
    CHECK_STATUS_FAIL_GOTO( status, final );

    fifo_depth = (int32_t)ceil((float)(tile_size[0] * tile_size[1]) / (float)hw_param.streamProcessorExecCount);
    max_vector_depth = hw_param.streamProcessorVectorSize;

    spinst = vsi_nn_sp_max_axis2_inst(ctx, fifo_depth, max_vector_depth);

    status = vxSetParameterByIndex( node, (uint32_t)index, (vx_reference)spinst->sp );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return status;
}

vsi_nn_kernel_node_t vsi_nn_sp_max_axis2_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1
    )
{
    const int32_t spInitInstsNum = 4;
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[7];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;
    uint32_t f32_min = 0xff800000;
    float flt_min = *(float*)&f32_min;
    float input_scale = vsi_nn_get_tensor_scale(input);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);
    clamp_min = clamp_min * input_scale;
    clamp_max = clamp_max * input_scale;

    /* init inst0: r2 = -INF */
    status = vsi_nn_sp_move_constant(&sp_insts_param[0], flt_min, VSI_NN_SP_SR2);
    /* init inst1: r10 = 0 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 0, VSI_NN_SP_SR10);
    /* init inst2: r4 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[2], 1, VSI_NN_SP_SR4);
    /* init inst3: nop */
    status |= vsi_nn_sp_nop(&sp_insts_param[3]);
    CHECK_STATUS_FAIL_GOTO(status, final);

    /* loop inst0: r1 = clamp(r3 * in, r6, r7) | r2 = v11 + r10 | r9 = r2 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[4], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_VR11, VSI_NN_SP_SR10, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[4], VSI_NN_SP_SR2, VSI_NN_SP_SR9);
    /* loop inst1: r8 = r1 * r4 | r5 = r1 - r2 | v11 = r5 ? r8 : r9 */
    status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR3, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_sub(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[5], VSI_NN_SP_SR5, VSI_NN_SP_SR8, VSI_NN_SP_VR11);
    /* loop inst2: out = r1 */
    status |= vsi_nn_sp_move(&sp_insts_param[6], VSI_NN_SP_SR1, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    attr.flush_cycle_num = 7;

    attr.ignored_leading_outputs = 1;
    attr.ignored_leading_v11_rd = 5;
    attr.ignored_leading_v11_wr = 2;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output0->t;
    outputs_tensor[1] = output1->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

final:

    if (node)
    {
        vxAssignNodeQueryCallback(node, max_axis2_query);
    }

    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_spinst_t * vsi_nn_sp_exp_y_direction_inst
    (
        vx_context                context,
        int32_t                   fifo_depth,
        int32_t                   max_vector_depth
    )
{
    vsi_status status = VSI_FAILURE;
    const int32_t spInitInstsNum = 2;
    const int32_t spLoopInstsNum = fifo_depth > 3 ? 4 : 8;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;
    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[10];
    vsi_nn_spinst_attr_t attr;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* init inst0: r8 = 0 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR8);
    /* init inst1: r9 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 1, VSI_NN_SP_SR9);
    CHECK_STATUS_FAIL_GOTO(status, final);

    if (fifo_depth > 3)
    {
        /* loop inst0: r2 = in - v11 | v11 = v11 */
        status  = vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        /* loop inst1: r8 = v12 * r9 | r7 = r4 + r6 | out = r7 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_VR12, VSI_NN_SP_SR9, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
        status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR7, VSI_NN_SP_SROUT);
        /* loop inst2: r6 = r5 * r2 | v12 = r7 + r8 | r4 = r3 */
        status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
        status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_SR7, VSI_NN_SP_SR8, VSI_NN_SP_VR12);
        status |= vsi_nn_sp_move(&sp_insts_param[4], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
        /* loop inst3: r1 = setup(r2) | r5 = pwlMul * pwlMul | r2 = pwlAdd + pwlAdd | r1 = r3*/
        status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[5], VSI_NN_SP_SR2, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_sub(&sp_insts_param[5], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 18;

        attr.ignored_leading_outputs = 4;
        attr.ignored_leading_v11_rd = 0;
        attr.ignored_leading_v11_wr = 0;
        attr.ignored_leading_v12_rd = fifo_depth + 3;
        attr.ignored_leading_v12_wr = 4;

        attr.num_of_v12_rd_in_flush_cycle = 4;
        attr.num_of_v12_wr_in_flush_cycle = 5;
    }
    else
    {
        /* loop inst0: r2 = in - v11 | v11 = v11 */
        status  = vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        /* loop inst1: r6 = r5 * r2 | r4 = r3 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
        status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
        /* loop inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[4]);
        /* loop inst3: r1 = setup(r2) */
        status  = vsi_nn_sp_pwl_setup0(&sp_insts_param[5], VSI_NN_SP_SR2, VSI_NN_SP_SR1);
        /* loop inst4: r8 = v12 * r9 | r7 = r4 + r6 */
        status |= vsi_nn_sp_mul(&sp_insts_param[6], VSI_NN_SP_VR12, VSI_NN_SP_SR9, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_add(&sp_insts_param[6], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
        /* loop inst5: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[7]);
        /* loop inst6: r5 = pwlMul * pwlMul | r2 = pwlAdd + pwlAdd | r1 = r3*/
        status |= vsi_nn_sp_mul(&sp_insts_param[8], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_sub(&sp_insts_param[8], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[8], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
        /* loop inst7: v12 = r7 + r8 | out = r7 */
        status |= vsi_nn_sp_add(&sp_insts_param[9], VSI_NN_SP_SR7, VSI_NN_SP_SR8, VSI_NN_SP_VR12);
        status |= vsi_nn_sp_move(&sp_insts_param[9], VSI_NN_SP_SR7, VSI_NN_SP_SROUT);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_outputs = 1;
        attr.ignored_leading_v11_rd = 0;
        attr.ignored_leading_v11_wr = 0;
        attr.ignored_leading_v12_rd = fifo_depth + 1;
        attr.ignored_leading_v12_wr = 1;

        attr.num_of_v12_rd_in_flush_cycle = 2;
        attr.num_of_v12_wr_in_flush_cycle = 2;

        attr.flush_cycle_num = 15;
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst_by_context(context);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    return spinst;
}

DEF_SP_KERNEL_QUERY(softmax_z_direction_exp_query)
    (
    vsi_nn_kernel_node_t        node
    )
{
    vsi_status status = VSI_FAILURE;
    vx_size index = 0;
    vx_size tile_size[2] = {0};
    vsi_nn_spinst_t *spinst = NULL;
    int32_t fifo_depth = 0;
    int32_t max_vector_depth = 0;
    vx_context  ctx = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_ext2_t hw_param;

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_ext2_t));
    status = vxQueryHardwareCaps(ctx, (vx_hardware_caps_params_t*)(&hw_param), sizeof(vx_hardware_caps_params_ext2_t));
    CHECK_STATUS_FAIL_GOTO( status, final );

    status = vxQueryNode(node, VX_NODE_SWTILING_TILE_XY, tile_size, sizeof(tile_size));
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vxQueryNode(node, VX_NODE_SPINST_INDEX, &index, sizeof(index));
    CHECK_STATUS_FAIL_GOTO( status, final );

    fifo_depth = (int32_t)ceil((float)(tile_size[0] * tile_size[1])/ (float)hw_param.streamProcessorExecCount);
    max_vector_depth = hw_param.streamProcessorVectorSize;

    spinst = vsi_nn_sp_exp_y_direction_inst(ctx, fifo_depth, max_vector_depth);

    status = vxSetParameterByIndex( node, (uint32_t)index, (vx_reference)spinst->sp );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return status;
}

vsi_nn_kernel_node_t vsi_nn_sp_softmax_z_direction_exp_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        float                         beta
    )
{
    const int32_t spInitInstsNum = 2;
    const int32_t spLoopInstsNum = 4;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[6];
    vsi_nn_spinst_attr_t attr;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;
    int32_t fifo_depth = 4;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* init inst0: r8 = 0 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR8);
    /* init inst1: r9 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 1, VSI_NN_SP_SR9);
    CHECK_STATUS_FAIL_GOTO(status, final);

    /* loop inst0: r2 = in - v11 | v11 = v11 */
    status  = vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
    /* loop inst1: r8 = v12 * r9 | r7 = r4 + r6 | out = r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_VR12, VSI_NN_SP_SR9, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR7, VSI_NN_SP_SROUT);
    /* loop inst2: r6 = r5 * r2 | v12 = r7 + r8 | r4 = r3 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_SR7, VSI_NN_SP_SR8, VSI_NN_SP_VR12);
    status |= vsi_nn_sp_move(&sp_insts_param[4], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
    /* loop inst3: r1 = setup(r2) | r5 = pwlMul * pwlMul | r2 = pwlAdd + pwlAdd | r1 = r3*/
    status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[5], VSI_NN_SP_SR2, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[5], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.flush_cycle_num = 18;

    attr.ignored_leading_outputs = 4;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v11_wr = 0;
    attr.ignored_leading_v12_rd = fifo_depth + 3;
    attr.ignored_leading_v12_wr = 4;

    attr.num_of_v12_rd_in_flush_cycle = 4;
    attr.num_of_v12_wr_in_flush_cycle = 5;

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    vx_lut_params.lut_function = VX_NN_ACTIVATION_CUSTOM;
    vx_lut_params.in_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);
    vx_lut_params.out_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);

    sp_lut_params.act_type = VSI_NN_SP_ACT_LINEAR_EXP;
    sp_lut_params.params[0] = beta;
    sp_lut_params.params[1] = 0;
    vsi_nn_sp_lut(vx_lut_params.in_lut, vx_lut_params.out_lut, &sp_lut_params);

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    outputs_tensor[0] = output0->t;
    outputs_tensor[1] = output1->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        &vx_lut_params);

final:
    if (node)
    {
        vxAssignNodeQueryCallback(node, softmax_z_direction_exp_query);
    }

    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    if (vx_lut_params.in_lut)
    {
        vxReleaseLUT(&vx_lut_params.in_lut);
        vx_lut_params.in_lut = NULL;
    }

    if (vx_lut_params.out_lut)
    {
        vxReleaseLUT(&vx_lut_params.out_lut);
        vx_lut_params.out_lut = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
}
vsi_nn_kernel_node_t vsi_nn_sp_rcp_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input,
        vsi_nn_tensor_t * output,
        float             output_scale
    )
{
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* loop inst0: r1 = pwlSetup(v12) | r5 = pwlMul() | r2 = pwlAdd() | r8 = r1 */
    status  = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR8);
    /* loop inst1: r6 = r5 * r2 | r7 = r4 + r6 | r4 = r8 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR4, VSI_NN_SP_SR8);
    /* loop inst1: v12 = r7 * r3 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR7, VSI_NN_SP_SR3, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_v12_wr = 4;
    attr.ignored_leading_v12_rd = 0;
    attr.flush_cycle_num = 14;

    attr.num_of_v12_wr_in_flush_cycle = 5;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.0f / output_scale);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output->t;

    vx_lut_params.lut_function = VX_NN_ACTIVATION_CUSTOM;
    vx_lut_params.in_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);
    vx_lut_params.out_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);

    sp_lut_params.act_type = VSI_NN_SP_ACT_RCP;
    vsi_nn_sp_lut(vx_lut_params.in_lut, vx_lut_params.out_lut, &sp_lut_params);

    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        &vx_lut_params);

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    if (vx_lut_params.in_lut)
    {
        vxReleaseLUT(&vx_lut_params.in_lut);
        vx_lut_params.in_lut = NULL;
    }
    if (vx_lut_params.out_lut)
    {
        vxReleaseLUT(&vx_lut_params.out_lut);
        vx_lut_params.out_lut = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_spinst_t * vsi_nn_sp_times_inst
    (
        vx_context                context,
        int32_t                   fifo_depth,
        int32_t                   max_vector_depth
    )
{
    vsi_status status = VSI_FAILURE;
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = fifo_depth > 4 ? 1 : fifo_depth > 1 ? 3 : 5;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;
    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    if (fifo_depth > 4)
    {
        /* loop inst0: out = v12 * in | v12 = v12 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SRIN, VSI_NN_SP_SROUT);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_VR12);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (fifo_depth > 1)
    {
        /* loop inst0: out = v12 * in | v12 = v12 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SRIN, VSI_NN_SP_SROUT);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_VR12);
        /* loop inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[1]);
        /* loop inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[2]);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        /* loop inst0: out = v12 * in | v12 = v12 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SRIN, VSI_NN_SP_SROUT);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_VR12);
        /* loop inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[1]);
        /* loop inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[2]);
        /* loop inst3: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[3]);
        /* loop inst4: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[4]);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    attr.flush_cycle_num = 0;

    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v11_wr = 0;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst_by_context(context);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    return spinst;
}

DEF_SP_KERNEL_QUERY(times_query)
    (
    vsi_nn_kernel_node_t        node
    )
{
    vsi_status status = VSI_FAILURE;
    vx_size index = 0;
    vx_size tile_size[2] = {0};
    vsi_nn_spinst_t *spinst = NULL;
    int32_t fifo_depth = 0;
    int32_t max_vector_depth = 0;
    vx_context  ctx = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_ext2_t hw_param;

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_ext2_t));
    status = vxQueryHardwareCaps(ctx, (vx_hardware_caps_params_t*)(&hw_param), sizeof(vx_hardware_caps_params_ext2_t));
    CHECK_STATUS_FAIL_GOTO( status, final );

    status = vxQueryNode(node, VX_NODE_SWTILING_TILE_XY, tile_size, sizeof(tile_size));
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vxQueryNode(node, VX_NODE_SPINST_INDEX, &index, sizeof(index));
    CHECK_STATUS_FAIL_GOTO( status, final );

    fifo_depth = (int32_t)ceil((float)(tile_size[0] * tile_size[1]) / (float)hw_param.streamProcessorExecCount);
    max_vector_depth = hw_param.streamProcessorVectorSize;

    spinst = vsi_nn_sp_times_inst(ctx, fifo_depth, max_vector_depth);

    status = vxSetParameterByIndex( node, (uint32_t)index, (vx_reference)spinst->sp );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return status;
}

vsi_nn_kernel_node_t vsi_nn_sp_softmax_z_direction_times_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL, NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;
    int32_t fifo_depth = 5;

    vsi_nn_spinst_t *spinst = NULL;

    spinst = vsi_nn_sp_times_inst(graph->ctx->c, fifo_depth, max_vector_depth);

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    outputs_tensor[0] = output->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

    if (node)
    {
        vxAssignNodeQueryCallback(node, times_query);
    }

    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

/*
** This program requires sum operation in the z dimension.
** Instead of using the SUM Engine, the sum needs to be performed
** by Stream Processor instructions.
*/
vsi_nn_kernel_node_t softmax_z_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[3] = {NULL};
    vsi_nn_tensor_t * output_tensor[2] = {NULL};
    int32_t axis = 2;
    float beta = vsi_nn_kernel_param_get_float32( params, "beta" );
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.is_dummy = TRUE;
    attr.size[axis] = 1;
    dummy_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[2] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    output_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[0], "Create tensor fail.", final );
    output_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[1], "Create tensor fail.", final );

    node = vsi_nn_sp_max_axis2_node(graph, inputs[0], output_tensor[0], dummy_tensor[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create sp_max_axis2 fail.", final );
    node = vsi_nn_sp_softmax_z_direction_exp_node(graph, output_tensor[0], dummy_tensor[0],
        output_tensor[1], dummy_tensor[1], beta);
    CHECK_PTR_FAIL_GOTO( node, "Create exp_y_direction fail.", final );
    node = vsi_nn_sp_rcp_node(graph, dummy_tensor[1], dummy_tensor[2], output_scale);
    CHECK_PTR_FAIL_GOTO( node, "Create sp_rcp fail.", final );
    node = vsi_nn_sp_softmax_z_direction_times_node(graph, output_tensor[1], dummy_tensor[2], outputs[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create softmax_times fail.", final );

final:
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(output_tensor[0]);
    vsi_safe_release_tensor(output_tensor[1]);

    return node;
} /* softmax_z_direction() */

#endif
