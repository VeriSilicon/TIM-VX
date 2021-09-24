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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/** Comparisons Kernel internal type */
typedef enum
{
    COMP_GREAT = VSI_NN_RELATIONAL_OPS_GREAT,
    COMP_GREAT_EQUAL = VSI_NN_RELATIONAL_OPS_GREAT_EQUAL,
    COMP_LESS = VSI_NN_RELATIONAL_OPS_LESS,
    COMP_LESS_EQUAL = VSI_NN_RELATIONAL_OPS_LESS_EQUAL,
    COMP_NOT_EQUAL = VSI_NN_RELATIONAL_OPS_NOT_EQUAL,
    COMP_EQUAL = VSI_NN_RELATIONAL_OPS_EQUAL,
} relational_type_e;

/*
 * Define kernel meta.
 */
#define HASH_COMPARISONS_KEY(_type, _input0_type, _input1_type, _output_type, _image_2d) \
    ((_type << 28) | (_input0_type << 20) | (_input1_type << 12)  | (_output_type << 2) | (_image_2d))

#define KERNEL_SOURCE_2D    "relational_ops_2d",
#define KERNEL_SOURCE_3D    "relational_ops_3d",

#define HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, SRC0_TYPE, SRC1_TYPE) \
    CVIVANTE_NAMESPACE("evis."#FUNC_NAME"_"#SRC0_TYPE#SRC1_TYPE"toBOOL8")

#define COMPARISONS_KERNELS(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 0), \
        HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, SRC0_TYPE, SRC1_TYPE), \
        SOURCE },

#define HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, SRC0_TYPE, SRC1_TYPE) \
    CVIVANTE_NAMESPACE("evis."#FUNC_NAME"_"#SRC0_TYPE#SRC1_TYPE"toBOOL8_2D")

#define COMPARISONS_KERNELS_2D(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 1), \
        HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, SRC0_TYPE, SRC1_TYPE), \
        SOURCE },

#define COMPARISONS_KERNELS_INT8(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 0), \
        HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, I8, I8), \
        SOURCE },

#define COMPARISONS_KERNELS_INT8_2D(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 1), \
        HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, I8, I8), \
        SOURCE },

#define COMPARISONS_KERNELS_HALF(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 0), \
        HASH_COMPARISONS_SH_KERNEL_NAME(FUNC_NAME, F16, F16), \
        SOURCE },

#define COMPARISONS_KERNELS_HALF_2D(FUNC_NAME, TYPE, SRC0_TYPE, SRC1_TYPE, SOURCE) \
    {   HASH_COMPARISONS_KEY(TYPE, SRC0_TYPE, SRC1_TYPE, BOOL8, 1), \
        HASH_COMPARISONS_SH_KERNEL_2D_NAME(FUNC_NAME, F16, F16), \
        SOURCE },

#define LESS_OP              less
#define GREAT_OP             great
#define LESS_EQUAL_OP        less_equal
#define GREAT_EQUAL_OP       great_equal
#define EQUAL_OP             equal
#define NOT_EQUAL_OP         not_equal

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } _comparisons_evis_kernel_map[] =
{
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, F16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, F16,   U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, F16,   I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, I16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, I16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, U8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, I8,    I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_OP, COMP_LESS, I8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_HALF(LESS_OP, COMP_LESS, BF16,  BF16,  KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_INT8(LESS_OP, COMP_LESS, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, F16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, F16,   U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, F16,   I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, I16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, I16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, U8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, I8,    I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_OP, COMP_LESS, I8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_HALF_2D(LESS_OP, COMP_LESS, BF16,  BF16,  KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_INT8_2D(LESS_OP, COMP_LESS, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, F16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, F16,   U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, F16,   I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, I16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, I16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, U8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, I8,    I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_OP, COMP_GREAT, I8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_HALF(GREAT_OP, COMP_GREAT, BF16,  BF16,  KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_INT8(GREAT_OP, COMP_GREAT, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, F16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, F16,   U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, F16,   I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, I16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, I16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, U8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, I8,    I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_OP, COMP_GREAT, I8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_HALF_2D(GREAT_OP, COMP_GREAT, BF16,  BF16,  KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_INT8_2D(GREAT_OP, COMP_GREAT, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F16,   U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, F16,   I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, I16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, I16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, U8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, I8,    I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(LESS_EQUAL_OP,  COMP_LESS_EQUAL, I8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_HALF(LESS_EQUAL_OP,  COMP_LESS_EQUAL, BF16,  BF16,  KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_INT8(LESS_EQUAL_OP,  COMP_LESS_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, F16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, F16,   U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, F16,   I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, I16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, I16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, U8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, I8,    I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, I8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_HALF_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, BF16,  BF16,  KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_INT8_2D(LESS_EQUAL_OP, COMP_LESS_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, U8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I8,    I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_HALF(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, BF16,  BF16,  KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_INT8(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, F16,   I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, U8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I8,    I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, I8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_HALF_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, BF16,  BF16,  KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_INT8_2D(GREAT_EQUAL_OP, COMP_GREAT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, F16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, F16,   U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, F16,   I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, I16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, I16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, U8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, I8,    I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(EQUAL_OP, COMP_EQUAL, I8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_HALF(EQUAL_OP, COMP_EQUAL, BF16,  BF16,  KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_INT8(EQUAL_OP, COMP_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, F16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, F16,   U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, F16,   I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, I16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, I16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, U8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, I8,    I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(EQUAL_OP, COMP_EQUAL, I8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_HALF_2D(EQUAL_OP, COMP_EQUAL, BF16,  BF16,  KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_INT8_2D(EQUAL_OP, COMP_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)

    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, I16,   I16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, I16,   F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, U8,    U8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, U8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, I8,    I8,    KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS(NOT_EQUAL_OP, COMP_NOT_EQUAL, I8,    F16,   KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_HALF(NOT_EQUAL_OP, COMP_NOT_EQUAL, BF16,  BF16,  KERNEL_SOURCE_3D)
    COMPARISONS_KERNELS_INT8(NOT_EQUAL_OP, COMP_NOT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_3D)

    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, F16,   I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, I16,   I16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, I16,   F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, U8,    U8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, U8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, I8,    I8,    KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, I8,    F16,   KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_HALF_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, BF16,  BF16,  KERNEL_SOURCE_2D)
    COMPARISONS_KERNELS_INT8_2D(NOT_EQUAL_OP, COMP_NOT_EQUAL, BOOL8, BOOL8, KERNEL_SOURCE_2D)
};

#undef LESS_OP
#undef GREAT_OP
#undef LESS_EQUAL_OP
#undef GREAT_EQUAL_OP
#undef EQUAL_OP
#undef NOT_EQUAL_OP

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_comparisons_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * attr[3]   = { NULL, NULL, NULL };
    vsi_size_array_t * out_shape             = NULL;
    float    input0Scale                    = 1.0f;
    float    input0Tail                     = 0;
    float    input1Scale                    = 1.0f;
    float    input1Tail                     = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_shape  = attr[2]->shape;

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[0]->dfp.fl;
        if (fl > 0)
        {
            input0Scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input0Scale = (float)((int64_t)1 << -fl);
        }
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input0Scale  = attr[0]->asymm.scale;
        input0Tail = 0 - attr[0]->asymm.zero_point * input0Scale;
    }

    if( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[1]->dfp.fl;
        if (fl > 0)
        {
            input1Scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input1Scale = (float)((int64_t)1 << -fl);
        }
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input1Scale  = attr[1]->asymm.scale;
        input1Tail = 0 - attr[1]->asymm.zero_point * input1Scale;
    }

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    if (1)
    {
            gpu_dp_inst_t uniExtractInteger_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniDatatoFp32Part0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniDatatoFp32Part1_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniDatatoFp32Part0_4x4", &uniDatatoFp32Part0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniDatatoFp32Part1_4x4", &uniDatatoFp32Part1_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "input0Scale", &input0Scale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "input0Tail", &input0Tail );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "input1Scale", &input1Scale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "input1Tail", &input1Tail );
            CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    return status;
} /* _comparisons_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t operation,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype;
    vsi_nn_kernel_dtype_e input1_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    output_dtype = output_dtype == I8 ? BOOL8 : output_dtype;
    key = HASH_COMPARISONS_KEY( operation, input0_dtype, input1_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(_comparisons_evis_kernel_map); i ++ )
    {
        if( _comparisons_evis_kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(_comparisons_evis_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _comparisons_evis_kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _comparisons_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                _comparisons_evis_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _comparisons_evis_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */


static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_EVIS_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    int32_t operation = 0;

    if( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    operation = vsi_nn_kernel_param_get_int32( params, "operation" );

    image_2d = (outputs[0]->attr.dim_num == 2);
    status = _query_kernel( inputs, outputs, operation, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _EVIS_PARAM_NUM,
                    inputs, 2, outputs, 1 );

            status = vsi_nn_kernel_node_pass_param( node, node_params, _EVIS_PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

REGISTER_BACKEND_EVIS( relational_ops, _setup )

__END_DECLS
