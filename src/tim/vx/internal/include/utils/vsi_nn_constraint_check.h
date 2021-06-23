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
/** @file */
#ifndef _VSI_NN_CONSTRAINT_CHECK_H
#define _VSI_NN_CONSTRAINT_CHECK_H

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* short alias for dtype */
enum {
    D_NONE = VSI_NN_TYPE_NONE,
    D_I8 = VSI_NN_TYPE_INT8,
    D_I16 = VSI_NN_TYPE_INT16,
    D_I32 = VSI_NN_TYPE_INT32,
    D_I64 = VSI_NN_TYPE_INT64,
    D_U8 = VSI_NN_TYPE_UINT8,
    D_U16 = VSI_NN_TYPE_UINT16,
    D_U32 = VSI_NN_TYPE_UINT32,
    D_U64 = VSI_NN_TYPE_UINT64,
    D_F16 = VSI_NN_TYPE_FLOAT16,
    D_F32 = VSI_NN_TYPE_FLOAT32,
    D_F64 = VSI_NN_TYPE_FLOAT64,
    D_BF16 = VSI_NN_TYPE_BFLOAT16,
    D_BOOL8 = VSI_NN_TYPE_BOOL8
};

/* short alias for qtype */
enum {
    Q_SHIFT = 8,
    Q_NONE = VSI_NN_QNT_TYPE_NONE << Q_SHIFT,
    Q_DFP = VSI_NN_QNT_TYPE_DFP << Q_SHIFT,
    Q_ASYM = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC << Q_SHIFT,
    Q_SYM_PC = VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC << Q_SHIFT,
    Q_SYM = VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC << Q_SHIFT,
};

typedef struct {
    uint32_t reg_input_num;
    uint32_t reg_output_num;
    uint32_t io_types_item_size;
    uint32_t io_types_item_count;
    const void* types;
} op_constraint_reg_type;

vsi_bool is_const_tensor
    (
    const vsi_nn_tensor_t* tensor
    );

vsi_bool validate_op_io_types
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t** inputs,
    int inputs_num,
    vsi_nn_tensor_t** outputs,
    int outputs_num,
    const op_constraint_reg_type* op_constraint_reg,
    const char* name
    );

void print_op_io_types
    (
    const char* name,
    const op_constraint_reg_type* op_constraint_reg
    );

char* generate_op_io_types_desc
    (
    vsi_nn_tensor_t** inputs,
    int inputs_num,
    vsi_nn_tensor_t** outputs,
    int outputs_num
    );

void destroy_op_io_types_desc
    (
    char* desc
    );

vsi_bool is_item_in_array
    (
    const void* item,
    const void* items,
    int item_size,
    int item_count
    );

#define IO_TYPE(...) {{__VA_ARGS__}},
#define BEGIN_IO_TYPE_DECL(NAME, INPUT_COUNT, OUTPUT_COUNT)         \
enum { NAME##_INPUT_COUNT = INPUT_COUNT,                            \
    NAME##_OUTPUT_COUNT = OUTPUT_COUNT,                             \
    NAME##_IO_COUNT = NAME##_INPUT_COUNT + NAME##_OUTPUT_COUNT};    \
static const struct {int types[NAME##_IO_COUNT];}         \
NAME##_supported_io_types[] = {

#define DECL_OP_CONSTRAINT_REG(NAME) \
static const op_constraint_reg_type NAME##_REG = {  \
    NAME##_INPUT_COUNT,                             \
    NAME##_OUTPUT_COUNT,                            \
    sizeof(NAME##_supported_io_types[0]),           \
    _cnt_of_array(NAME##_supported_io_types),       \
    NAME##_supported_io_types                       \
}

#ifdef OUTPUT_OP_CONSTRAINT
#define END_IO_TYPE_DECL(NAME) }; \
    DECL_OP_CONSTRAINT_REG(NAME);
    print_op_io_types(#NAME, NAME##_IO_COUNT, NAME##_supported_io_types, \
        _cnt_of_array(NAME##_supported_io_types));
#else
#define END_IO_TYPE_DECL(NAME) };                   \
DECL_OP_CONSTRAINT_REG(NAME);
#endif

#define VALIDATE_OP_IO_TYPES(NAME, SELF, INPUTS, INPUTS_NUM, OUTPUTS, OUTPUTS_NUM) \
    validate_op_io_types(SELF, INPUTS, INPUTS_NUM, OUTPUTS, OUTPUTS_NUM, \
        &NAME##_REG, #NAME)

#ifdef __cplusplus
}
#endif

#endif
