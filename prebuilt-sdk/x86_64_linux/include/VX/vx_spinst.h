/****************************************************************************
*
*    Copyright 2017 - 2021 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef _VX_SPINST_H_
#define _VX_SPINST_H_

#ifdef  __cplusplus
extern "C" {
#endif

typedef enum _vx_sp_inst_type_e
{
    VX_SP_INST_TYPE_FADD,
    VX_SP_INST_TYPE_FMULT,
    VX_SP_INST_TYPE_MOVE,
    VX_SP_INST_TYPE_PWL,

    VX_SP_INST_TYPE_COUNT,
}
vx_sp_inst_type_e;

typedef enum _vx_sp_inst_type_fadd_e
{
    VX_SP_INST_TYPE_FADD_IDLE,   // FADD-IDLE
    VX_SP_INST_TYPE_FADD_ADD,    // dst = src0 + src1
    VX_SP_INST_TYPE_FADD_SUB,    // dst = src0 - src1

    VX_SP_INST_TYPE_FADD_COUNT,
}
vx_sp_inst_type_fadd_e;

typedef enum _vx_sp_inst_type_fmult_e
{
    VX_SP_INST_TYPE_FMULT_IDLE,       /* FMULT-IDLE */
    VX_SP_INST_TYPE_FMULT_MUL,        /* dst = src0 * src1 */
    VX_SP_INST_TYPE_FMULT_MUL_CLAMP,  /* dst = clamp (src0, src1, R6, R7) */

    VX_SP_INST_TYPE_FMULT_COUNT,
}
vx_sp_inst_type_fmult_e;

typedef enum _vx_sp_inst_type_move_e
{
    VX_SP_INST_TYPE_MOVE_IDLE,
    VX_SP_INST_TYPE_MOVE_MOVE,  // dst = src1
    VX_SP_INST_TYPE_MOVE_SEL0,  // dst = (src0 > 0) ? src1[0] : src1[1]
    VX_SP_INST_TYPE_MOVE_SEL1,  // dst = (src0 > 0) ? src1 : FA-src0  // use FA's SRC0
    VX_SP_INST_TYPE_MOVE_IMMD,  // dst = Constant assign immmediate
    VX_SP_INST_TYPE_MOVE_ABS,   // dst = abs(src1)

    VX_SP_INST_TYPE_MOVE_COUNT,
}
vx_sp_inst_type_move_e;

typedef enum _vx_sp_inst_type_pwl_e
{
    VX_SP_INST_TYPE_PWL_IDLE,
    VX_SP_INST_TYPE_PWL_SETUP_0,  /* PWL ID = 0 */
    VX_SP_INST_TYPE_PWL_SETUP_1,  /* Sigmode() */
    VX_SP_INST_TYPE_PWL_SETUP_2,  /* Tanh() */

    VX_SP_INST_TYPE_PWL_COUNT,
}
vx_sp_inst_type_pwl_e;

typedef enum _vx_sp_inst_src_dst_e
{
    VX_SP_INST_SPINOUT,
    VX_SP_INST_SR1,
    VX_SP_INST_SR2,
    VX_SP_INST_SR3,
    VX_SP_INST_SR4,
    VX_SP_INST_SR5,
    VX_SP_INST_SR6,   /* nn_clamp_min */
    VX_SP_INST_SR7,   /* nn_clamp_max */
    VX_SP_INST_SR8,
    VX_SP_INST_SR9,
    VX_SP_INST_SR10,
    VX_SP_INST_VR11,
    VX_SP_INST_VR12,
    VX_SP_INST_VR13,
    VX_SP_INST_VR14,
    VX_SP_INST_SETUPOUT,   /* Input of PWL Mult and Add: FMInA, FMInB, FAInA, FAInB */
}
vx_sp_inst_src_dst_e;

typedef struct _vx_spinst_unit_param
{
    vx_enum         op;    /* vx_sp_inst_type_e */

    struct
    {
        vx_enum     op;    /* vx_sp_inst_type_fadd/fmult/move/pwl_e */

        struct
        {
            vx_uint8    src0;       /* vx_sp_inst_src_dst_e */
            vx_uint8    src1;       /* vx_sp_inst_src_dst_e */
            vx_uint8    dst;        /* vx_sp_inst_src_dst_e */
            vx_float32  constant;
        } var;

    } sub;

}
vx_spinst_unit_param;

/**********************************************************************************************/

typedef enum _vx_sp_attribute_e
{
    VX_SP_ATTRIBUTE_NONE,

    VX_SP_ATTRIBUTE_INPUT_TILE_MAPPING,
    VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_X,
    VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Y,
    VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Z,

    VX_SP_ATTRIBUTE_PROG_INIT_INSTR_NUM,
    VX_SP_ATTRIBUTE_PROG_LOOP_INSTR_NUM,
    VX_SP_ATTRIBUTE_PROG_COMPLETE_INSTR_NUM,
    VX_SP_ATTRIBUTE_PROG_ROUNDING_MODE,
    VX_SP_ATTRIBUTE_INPUT_SETUP,

    VX_SP_ATTRIBUTE_IGNORED_LEADING_OUTPUTS,
    VX_SP_ATTRIBUTE_FLUSH_CYCLE_NUM,
    VX_SP_ATTRIBUTE_IGNORED_LEADING_V11_WR,
    VX_SP_ATTRIBUTE_IGNORED_LEADING_V12_WR,
    VX_SP_ATTRIBUTE_IGNORED_LEADING_V11_RD,
    VX_SP_ATTRIBUTE_IGNORED_LEADING_V12_RD,

    VX_SP_ATTRIBUTE_CH0_POST_REDISTRIBUTE,
    VX_SP_ATTRIBUTE_CH1_POST_REDISTRIBUTE,
    VX_SP_ATTRIBUTE_V11_RESET_AT_START,
    VX_SP_ATTRIBUTE_V12_RESET_AT_START,
    VX_SP_ATTRIBUTE_V11_POP_CONFIG,
    VX_SP_ATTRIBUTE_V12_POP_CONFIG,
    VX_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT,
    VX_SP_ATTRIBUTE_IGNORED_LEADING_ACC_OUT,
    VX_SP_ATTRIBUTE_SUM_ENGINE_RESET,
    VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL,
    VX_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE,
    VX_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE,
    VX_SP_ATTRIBUTE_SUM_ENGINE_OP_SELECT,

    VX_SP_ATTRIBUTE_NUM_OF_ELEMENTS_PER_LOOP_PER_INPUT,

    VX_SP_ATTRIBUTE_NUM_OF_V11_RD_IN_FLUSH_CYCLE,
    VX_SP_ATTRIBUTE_NUM_OF_V12_RD_IN_FLUSH_CYCLE,
    VX_SP_ATTRIBUTE_NUM_OF_V11_WR_IN_FLUSH_CYCLE,
    VX_SP_ATTRIBUTE_NUM_OF_V12_WR_IN_FLUSH_CYCLE,

    VX_SP_ATTRIBUTE_GENERAL_COUNT,

    VX_SP_ATTRIBUTE_CONST0,     /* NN post multiplier    */
    VX_SP_ATTRIBUTE_CONST1,     /* NN neg pos multiplier */
    VX_SP_ATTRIBUTE_CONST2,     /* NN tensor add const   */
    VX_SP_ATTRIBUTE_CONST3,     /* NN clamp max          */
    VX_SP_ATTRIBUTE_CONST4,     /* NN clmap min          */

    VX_SP_ATTRIBUTE_CONST_COUNT,

    VX_SP_ATTRIBUTE_SPLIT_AXIS,
    VX_SP_ATTRIBUTE_SPLIT_MAX_SIZE,
    VX_SP_ATTRIBUTE_SPLIT_TILEX_EQUAL_INIMAGEX,

    VX_SP_ATTRIBUTE_NOT_MERGE_CONVSP,
    VX_SP_ATTRIBUTE_UPDATE_CONST0_TO_PCQ_COEF_TENSOR,
    VX_SP_ATTRIBUTE_RESHAPE_ARRAY, /* bit layout | output:24-29 | input3:18-23 | input2:12-17 | input1:6-11 | input0:0-5 | */
    VX_SP_ATTRIBUTE_ALIGN_SP_CORE_AXIS,
    VX_SP_ATTRIBUTE_KEEP_TILE_SIZE,

    VX_SP_ATTRIBUTE_TOTAL_COUNT,
}
vx_sp_attribute_e;

typedef enum _vx_sp_attribute_input_tile_mapping_e
{
    VX_SP_ATTRIBUTE_INPUT_TILE_MAPPING_XYMERGE,
    VX_SP_ATTRIBUTE_INPUT_TILE_MAPPING_YZMERGE,
}
vx_sp_attribute_input_tile_mapping_e;

typedef enum _vx_sp_attribute_output_collapse_e
{
    VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_DISABLED,
    VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_ENABLED,
}
vx_sp_attribute_output_collapse_e;

typedef enum _vx_sp_attribute_rounding_mode_e
{
    VX_SP_ATTRIBUTE_PROG_ROUNDING_MODE_RTNE,
    VX_SP_ATTRIBUTE_PROG_ROUNDING_MODE_STICKY,
}
vx_sp_attribute_rounding_mode_e;

typedef enum _vx_sp_attribute_input_setup_e
{
    VX_SP_ATTRIBUTE_INPUT_SETUP_SINGLE_INPUT,
    VX_SP_ATTRIBUTE_INPUT_SETUP_INTERLEAVE_TWO_INPUTS,
    VX_SP_ATTRIBUTE_INPUT_SETUP_V11,
    VX_SP_ATTRIBUTE_INPUT_SETUP_V12,
}
vx_sp_attribute_input_setup_e;

typedef enum _vx_sp_attribute_ch_post_redistribute_e
{
    VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_DISABLED,
    VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_SCALAR_GATHER,
    VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_VECTOR_GATHER,
    VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_VECTOR_SCATTER,
}
vx_sp_attribute_ch_post_redistribute_e;

typedef enum _vx_sp_attribute_v_reset_at_start_e
{
    VX_SP_ATTRIBUTE_V_RESET_AT_START_NONE,
    VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET,
}
vx_sp_attribute_v_reset_at_start_e;

typedef enum _vx_sp_attribute_v_pop_config_e
{
    VX_SP_ATTRIBUTE_V_POP_CONFIG_EVERY_READ,
    VX_SP_ATTRIBUTE_V_POP_CONFIG_EVERY_ROW,
}
vx_sp_attribute_v_pop_config_e;

typedef enum _vx_sp_attribute_accelerator_input_select_e
{
    VX_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT_FROM_OUTPUT,
    VX_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT_FROM_ACCLERATOR,
}
vx_sp_attribute_accelerator_input_select_e;

typedef enum _vx_sp_attribute_sum_engine_reset_e
{
    VX_SP_ATTRIBUTE_SUM_ENGINE_RESET_NONE,
    VX_SP_ATTRIBUTE_SUM_ENGINE_RESET_RESET,
}
vx_sp_attribute_sum_engine_reset_e;

typedef enum _vx_sp_attribute_sum_engine_control_e
{
    VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL_ACCUM_INTERNAL,
    VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL_ACCUM_1D,
    VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL_ACCUM_2D,
}
vx_sp_attribute_sum_engine_control_e;

typedef enum _vx_sp_attribute_sum_engine_num_ch_minus_one_e
{
    VX_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE_ONE_CH,
    VX_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE_TWO_CH,
}
vx_sp_attribute_sum_engine_num_ch_minus_one_e;

typedef enum _vx_sp_attribute_sum_engine_2d_accum_storage_e
{
    VX_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE_SAME,
    VX_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE_DIFFERENT,
}
vx_sp_attribute_sum_engine_2d_accum_storage_e;

typedef enum _vx_sp_attribute_sum_engine_op_select_e
{
    VX_SP_ATTRIBUTE_SUM_ENGINE_SUM_OP,
    VX_SP_ATTRIBUTE_SUM_ENGINE_MAX_OP
} vx_sp_attribute_sum_engine_op_select_e;

typedef enum _vx_sp_attribute_reshape_e
{
    VX_SP_ATTRIBUTE_RESHAPE_CHW2CHW = 0x00,
    VX_SP_ATTRIBUTE_RESHAPE_CHW2WHC = 0x06,
    VX_SP_ATTRIBUTE_RESHAPE_CHW2WCH = 0x09,
    VX_SP_ATTRIBUTE_RESHAPE_CHW2HWC = 0x12,
    VX_SP_ATTRIBUTE_RESHAPE_CHW2HCW = 0x18,
    VX_SP_ATTRIBUTE_RESHAPE_CHW2CWH = 0x21,
}
vx_sp_attribute_reshape_e;

typedef enum _vx_sp_attribute_split_axis_e
{
    VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_X,
    VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_Y,
    VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_Z,
    VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_XY,
    VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_YZ,
    VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_XYZ,
}
vx_sp_attribute_split_axis_e;

typedef enum _vx_sp_attribute_tile_align_sp_core_e
{
    VX_SP_ATTRIBUTE_TILE_ALIGN_SP_CORE_NONE = 0,
    VX_SP_ATTRIBUTE_TILE_ALIGN_SP_CORE_WITH_AXIS_X,
    VX_SP_ATTRIBUTE_TILE_ALIGN_SP_CORE_WITH_AXIS_Y,
    VX_SP_ATTRIBUTE_TILE_ALIGN_SP_CORE_WITH_AXIS_XY,
}
vx_sp_attribute_tile_align_sp_core_e;

typedef enum _vx_sp_attribute_keep_tile_size_e
{
    VX_SP_ATTRIBUTE_KEEP_TILE_SIZE_NONE = 0,
    VX_SP_ATTRIBUTE_KEEP_TILE_SIZE_WITH_AXIS_X,
    VX_SP_ATTRIBUTE_KEEP_TILE_SIZE_WITH_AXIS_Y,
    VX_SP_ATTRIBUTE_KEEP_TILE_SIZE_WITH_AXIS_XY,
}
vx_sp_attribute_keep_tile_size_e;

/**********************************************************************************************/

/*! \brief Creates an external reference to a spinst data.
 * \param [in] context The reference to the implementation context.
 * \return A spinst data reference.
 * \Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_object_spinst
 */
VX_API_ENTRY vx_spinst VX_API_CALL vxCreateSPINST(
    vx_context          context
    );

/*! \brief Releases a reference to a external spinst object.
 * The object may not be garbage collected until its total reference count is zero.
 * \param [in] spinst_obj The pointer to the spinst data to release.
 * \post After returning from this function the reference is zeroed.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; all other values indicate failure
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 * \ingroup group_object_spinst
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseSPINST(
    vx_spinst            *spinst_obj
    );

/*! \brief Add a instruction to spinst object.
 * \param [in] spinst_obj The reference to the spinst object.
 * \param [in] inst_unit_array The units of one instruction. Use a <tt>\ref vx_spinst_unit_param</tt>.
 * \param [in] inst_unit_count The count of instruction units.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If data is not a <tt>\ref spinst_obj</tt>.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of parameters is incorrect.
 * \retval VX_ERROR_NO_MEMORY If fail to allocate internal instruction memory.
 * \ingroup group_object_spinst
 */
VX_API_ENTRY vx_status VX_API_CALL vxAddOneInstToSPINST(
    vx_spinst                 spinst_obj,
    vx_spinst_unit_param*     inst_unit_array,
    vx_uint8                  inst_unit_count
    );

/*! \brief Set various attributes of a spinst data.
 * \param [in] spinst_obj The reference to the vx_spinst object to set.
 * \param [in] attribute The attribute to set. Use a <tt>\ref vx_sp_attribute_e</tt>.
 * \param [in] value The value of attribute.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If data is not a <tt>\ref vx_spinst</tt>.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of attribute is incorrect.
 * \ingroup group_object_spinst
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetAttributeToSPINST(
    vx_spinst          spinst_obj,
    vx_enum            attribute,
    vx_uint32          value
    );

VX_API_ENTRY vx_status VX_API_CALL vxGetAttributeToSPINST(
    vx_spinst          spinst_obj,
    vx_enum            attribute,
    vx_uint32* value
);

#ifdef  __cplusplus
}
#endif

#endif
