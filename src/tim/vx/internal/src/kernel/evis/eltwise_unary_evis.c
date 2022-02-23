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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/** Unary Kernel internal type */
typedef enum
{
    UNARY_SIN,
    UNARY_COS,
    UNARY_EXP,
    UNARY_LOG,
    UNARY_ELU,
    UNARY_NEG,
    UNARY_HSIGMOID,
    UNARY_MISH,
    UNARY_ROUND,
    UNARY_GELU,
    UNARY_HGELU,
} unary_type_e;

/*
 * Define kernel meta.
 */
#define HASH_UNARY_KEY(_type, _input_type, _output_type, _image_2d) \
    ((_type << 20) | (_input_type << 12) | (_output_type << 4) | (_image_2d))

#define KERNEL_SOURCE_2D    "eltwise_unary_2d",
#define KERNEL_SOURCE_3D    "eltwise_unary_3d",

#define HASH_UNARY_SH_KERNEL_NAME(FUNC_NAME, SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis."#FUNC_NAME"_"#SRC_TYPE"to"#DST_TYPE)

#define TENSOR_UNARY_KERNELS(FUNC_NAME, TYPE, SRC_TYPE, OUT_TYPE, SOURCE) \
    {   HASH_UNARY_KEY(TYPE, SRC_TYPE, OUT_TYPE, 0), \
        HASH_UNARY_SH_KERNEL_NAME(FUNC_NAME, SRC_TYPE, OUT_TYPE), \
        SOURCE },

#define HASH_UNARY_SH_KERNEL_2D_NAME(FUNC_NAME, SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis."#FUNC_NAME"_"#SRC_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_UNARY_KERNELS_2D(FUNC_NAME, TYPE, SRC_TYPE, OUT_TYPE, SOURCE) \
    {   HASH_UNARY_KEY(TYPE, SRC_TYPE, OUT_TYPE, 1), \
        HASH_UNARY_SH_KERNEL_2D_NAME(FUNC_NAME, SRC_TYPE, OUT_TYPE), \
        SOURCE },

#define SIN_OPERATION           sin
#define COS_OPERATION           cos
#define EXP_OPERATION           exp
#define LOG_OPERATION           log
#define ELU_OPERATION           elu
#define NEG_OPERATION           neg
#define HSIGMOID_OPERATION      hard_sigmoid
#define MISH_OPERATION          mish
#define ROUND_OPERATION         round
#define GELU_OPERATION          gelu
#define HGELU_OPERATION         hard_gelu

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } _eltwise_unary_evis_kernel_map[] =
{
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(SIN_OPERATION, UNARY_SIN, BF16, BF16 , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(COS_OPERATION, UNARY_COS, BF16, BF16 , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(EXP_OPERATION, UNARY_EXP, BF16, BF16 , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(LOG_OPERATION, UNARY_LOG, BF16, BF16 , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ELU_OPERATION, UNARY_ELU, BF16, BF16 , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(NEG_OPERATION, UNARY_NEG, BF16, BF16 , KERNEL_SOURCE_3D)

    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(SIN_OPERATION, UNARY_SIN, BF16, BF16 , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(COS_OPERATION, UNARY_COS, BF16, BF16 , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(EXP_OPERATION, UNARY_EXP, BF16, BF16 , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(LOG_OPERATION, UNARY_LOG, BF16, BF16 , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ELU_OPERATION, UNARY_ELU, BF16, BF16 , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(NEG_OPERATION, UNARY_NEG, BF16, BF16 , KERNEL_SOURCE_2D)

    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(HSIGMOID_OPERATION, UNARY_HSIGMOID, BF16, BF16 , KERNEL_SOURCE_3D)

    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HSIGMOID_OPERATION, UNARY_HSIGMOID, BF16, BF16 , KERNEL_SOURCE_2D)

    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(MISH_OPERATION, UNARY_MISH, BF16, BF16 , KERNEL_SOURCE_3D)

    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(MISH_OPERATION, UNARY_MISH, BF16, BF16 , KERNEL_SOURCE_2D)

    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(ROUND_OPERATION, UNARY_ROUND, BF16, BF16 , KERNEL_SOURCE_3D)

    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(ROUND_OPERATION, UNARY_ROUND, BF16, BF16 , KERNEL_SOURCE_2D)

    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, F16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, F16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, F16,  U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, F16,  I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, I16,  I16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, I16,  F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, U8,   U8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, U8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, I8,   I8   , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, I8,   F16  , KERNEL_SOURCE_3D)
    TENSOR_UNARY_KERNELS(GELU_OPERATION, UNARY_GELU, BF16, BF16 , KERNEL_SOURCE_3D)

    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(GELU_OPERATION, UNARY_GELU, BF16, BF16 , KERNEL_SOURCE_2D)

    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, F16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, F16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, F16,  U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, F16,  I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, I16,  I16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, I16,  F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, U8,   U8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, U8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, I8,   I8   , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, I8,   F16  , KERNEL_SOURCE_2D)
    TENSOR_UNARY_KERNELS_2D(HGELU_OPERATION, UNARY_HGELU, BF16, BF16 , KERNEL_SOURCE_2D)
};

#undef SIN_OPERATION
#undef COS_OPERATION
#undef EXP_OPERATION
#undef LOG_OPERATION
#undef ELU_OPERATION
#undef NEG_OPERATION
#undef HSIGMOID_OPERATION
#undef MISH_OPERATION
#undef ROUND_OPERATION
#undef GELU_OPERATION
#undef HGELU_OPERATION

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define INPUT_FUNC_TYPE           (2)
#define INPUT_SCALAR_ALPHA        (3)
#define INPUT_SCALAR_BETA         (4)
#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_eltwise_unary_initializer)
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
    int32_t     type                        = 0;
    vsi_nn_kernel_tensor_attr_t * attr[2]   = { NULL, NULL };
    vsi_size_array_t * out_shape             = NULL;
    float    inputScale                     = 1.0f;
    float    inputTail                      = 0;
    float    outputScale                    = 1.0f;
    float    outputZP                       = 0;
    float    alpha                          = 0;
    float    beta                           = 0;
    uint32_t pack_key;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[INPUT_FUNC_TYPE], &type);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[INPUT_SCALAR_ALPHA], &alpha);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[INPUT_SCALAR_BETA], &beta);
    CHECK_STATUS_FAIL_GOTO(status, final );

    out_shape  = attr[1]->shape;

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[0]->dfp.fl;
        if (fl > 0)
        {
            inputScale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            inputScale = (float)((int64_t)1 << -fl);
        }
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        inputScale  = attr[0]->asymm.scale;
        inputTail = 0 - attr[0]->asymm.zero_point * inputScale;
    }

    if( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[1]->dfp.fl;
        if (fl > 0)
        {
            outputScale = (float)((int64_t)1 << fl);
        }
        else
        {
            outputScale = (float)1.0f / (float) ((int64_t)1 << -fl);
        }
    }
    else if( attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        outputScale = (float)1.0f / attr[1]->asymm.scale;
        outputZP     = (float)attr[1]->asymm.zero_point;
    }

#define _PACK_SELECT_KEY( TYPE, IN_TYPE, OUT_TYPE )    \
        (( TYPE << 24) | ( IN_TYPE << 16) | ( OUT_TYPE << 8))

    pack_key = _PACK_SELECT_KEY( type, attr[0]->dtype, attr[1]->dtype );

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

    switch( pack_key )
    {
        case _PACK_SELECT_KEY( UNARY_SIN, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_COS, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_EXP, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_LOG, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_ELU, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_NEG, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_HSIGMOID, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_MISH, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_ROUND, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_GELU, BF16, BF16 ):
        case _PACK_SELECT_KEY( UNARY_HGELU, BF16, BF16 ):
        {
            gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x01050004, 0x03070206, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x05050404, 0x07070606, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniExtractOddData_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x07050301, 0x07050301, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            }, GPU_DP_TYPE_16 };

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "alpha", &alpha );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "beta", &beta );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        {
            gpu_dp_inst_t uniExtractHalf8_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            }, GPU_DP_TYPE_16 };
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
                    "uniDatatoFp32Part0_4x4", &uniDatatoFp32Part0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniDatatoFp32Part1_4x4", &uniDatatoFp32Part1_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "inputScale", &inputScale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "inputTail", &inputTail );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "outputScale", &outputScale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "outputZP", &outputZP );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "alpha", &alpha );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "beta", &beta );

            if (attr[1]->dtype == F16)
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtract8Data_2x8", &uniExtractHalf8_2x8 );
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
            }
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _eltwise_unary_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t type,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int i;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_UNARY_KEY( type, input_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(_eltwise_unary_evis_kernel_map); i ++ )
    {
        if( _eltwise_unary_evis_kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(_eltwise_unary_evis_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _eltwise_unary_evis_kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _eltwise_unary_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                _eltwise_unary_evis_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _eltwise_unary_evis_kernel_map[i].source_name );
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
    vsi_nn_kernel_t             * kernel,
    const unary_type_e            unary_type
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* rs_tensors[2] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t new_rank = 0;
    vsi_bool ret = FALSE;
    float alpha = vsi_nn_kernel_param_get_float32( params, "alpha" );
    float beta = vsi_nn_kernel_param_get_float32( params, "beta" );

    ret = vsi_nn_kernel_optimize_element_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );
    if( ret )
    {
        rs_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shape, new_rank );
        rs_tensors[1] = vsi_nn_reshape_tensor( graph,
                outputs[0], shape, new_rank );
    }

    if( !vsi_nn_kernel_gpu_check_shape( rs_tensors[0]->attr.size,
                rs_tensors[0]->attr.dim_num ) )
    {
        goto OnError;
    }

    image_2d = (rs_tensors[0]->attr.dim_num == 2 || rs_tensors[0]->attr.size[2] == 1);
    status = _query_kernel( rs_tensors, &rs_tensors[1], unary_type, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if( node )
        {
            vsi_nn_kernel_node_pack_io( node_params, _CL_PARAM_NUM,
                    rs_tensors, 1, &rs_tensors[1], 1 );
            node_params[INPUT_FUNC_TYPE] = vsi_nn_kernel_scalar_create(
                    graph, I32, &unary_type );
            node_params[INPUT_SCALAR_ALPHA] = vsi_nn_kernel_scalar_create(
                    graph, F32, &alpha );
            node_params[INPUT_SCALAR_BETA] = vsi_nn_kernel_scalar_create(
                    graph, F32, &beta );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, OnError );
        }
    }

OnError:
    if (rs_tensors[0])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[0] );
    }

    if (rs_tensors[1])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[1] );
    }

    if (node_params[INPUT_FUNC_TYPE])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_FUNC_TYPE] );
    }

    if (node_params[INPUT_SCALAR_ALPHA])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_SCALAR_ALPHA] );
    }

    if (node_params[INPUT_SCALAR_BETA])
    {
        vsi_nn_kernel_scalar_release( &node_params[INPUT_SCALAR_BETA] );
    }

    return node;
} /* _setup() */

#define REGISTER_ELTWISE_UNARY_BACKEND_EVIS(KERNEL_NAME, UNARY_TYPE) \
    static vsi_nn_kernel_node_t _##KERNEL_NAME##_setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num, \
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ) \
    { \
        return _setup(graph, inputs, input_num, outputs, output_num, \
                params, kernel, UNARY_TYPE); \
    } \
    REGISTER_BACKEND_EVIS( KERNEL_NAME, _##KERNEL_NAME##_setup )

REGISTER_ELTWISE_UNARY_BACKEND_EVIS( sin, UNARY_SIN )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( cos, UNARY_COS )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( exp, UNARY_EXP )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( log, UNARY_LOG )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( elu, UNARY_ELU )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( neg, UNARY_NEG )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( hard_sigmoid, UNARY_HSIGMOID )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( mish, UNARY_MISH )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( round, UNARY_ROUND )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( gelu, UNARY_GELU )
REGISTER_ELTWISE_UNARY_BACKEND_EVIS( hard_gelu, UNARY_HGELU )

__END_DECLS
