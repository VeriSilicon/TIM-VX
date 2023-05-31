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
#ifndef _VSI_NN_OPS_H
#define _VSI_NN_OPS_H

/*------------------------------------
               Includes
  -----------------------------------*/
#include "vsi_nn_platform.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"

#if defined(__cplusplus)
extern "C"{
#endif

/*------------------------------------
                Types
  -----------------------------------*/

/**
 * Ovxlib operation type index.
 * Custom ID is start from VSI_NN_OP_CUSTOM_START.
 * Ovxlib internal id is start from VSI_NN_OP_INTERNAL_START.
 * @see include/interface/ops.def
 * @see include/custom/custom_ops.def
 * @see include/internal/internal_ops.def
 */
typedef int32_t VSI_PUBLIC_TYPE vsi_nn_op_t; enum
{
#define DEF_OP( NAME, ... ) VSI_NN_OP_##NAME,
    #include "interface/ops.def"
#undef DEF_OP
    VSI_NN_OP_NUM,
    VSI_NN_OP_NA = VSI_NN_OP_NUM,
    VSI_NN_OP_CLIENT = VSI_NN_OP_NA + 1,
    // add operation alias for compilation
    VSI_NN_OP_ELTWISEMAX = VSI_NN_OP_MAXIMUM,

    VSI_NN_OP_CUSTOM_START = 0x10000,
#define DEF_OP( NAME, ... ) VSI_NN_OP_##NAME,
    #include "custom/custom_ops.def"
#undef DEF_OP
    VSI_NN_OP_CUSTOM_END,
    VSI_NN_OP_CUSTOM_NUM = VSI_NN_OP_CUSTOM_END - VSI_NN_OP_CUSTOM_START - 1,

    VSI_NN_OP_INTERNAL_START = 0x10000000,
#define DEF_OP( NAME, ... ) VSI_NN_OP_##NAME,
    #include "internal/internal_ops.def"
#undef DEF_OP
    VSI_NN_OP_INTERNAL_END,
    VSI_NN_OP_INTERNAL_NUM = VSI_NN_OP_INTERNAL_END - VSI_NN_OP_INTERNAL_START - 1
};

/** Operation initialization function handler */
typedef vsi_status ( * vsi_nn_op_init_t )
    ( vsi_nn_node_t * );

/** Operation computation function handler */
typedef vsi_status ( * vsi_nn_op_compute_t )
    (
    vsi_nn_node_t *,
    vsi_nn_tensor_t **,
    vsi_nn_tensor_t **
    );

/** Operation deinitialization function handler */
typedef vsi_status ( * vsi_nn_op_deinit_t )
    ( vsi_nn_node_t * );

/** Operation validation function handler */
typedef vsi_bool ( * vsi_nn_op_check_t )
    (
    vsi_nn_node_t *,
    vsi_nn_tensor_t **,
    vsi_nn_tensor_t **
    );

/** Operation setup function handler */
typedef vsi_bool ( * vsi_nn_op_setup_t )
    (
    vsi_nn_node_t *,
    vsi_nn_tensor_t **,
    vsi_nn_tensor_t **
    );

/** Operation optimization function handler */
typedef vsi_status ( * vsi_nn_op_optimize_t )
    (
    vsi_nn_node_t *,
    vsi_nn_tensor_t **,
    vsi_nn_tensor_t **,
    vsi_nn_opt_direction_e
    );

/** Operation runtime interface. */
typedef struct _vsi_nn_op_proc
{
    vsi_nn_op_init_t     init;
    vsi_nn_op_compute_t  compute;
    vsi_nn_op_deinit_t   deinit;
    vsi_nn_op_check_t    check;
    vsi_nn_op_setup_t    setup;
    vsi_nn_op_optimize_t optimize;
    uint32_t            input_num;
    uint32_t            output_num;
} VSI_PUBLIC_TYPE vsi_nn_op_proc_t;

/*------------------------------------
              Functions
  -----------------------------------*/

OVXLIB_API vsi_status vsi_nn_op_common_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

OVXLIB_API vsi_status vsi_nn_op_common_deinit
    (
    vsi_nn_node_t * self
    );

OVXLIB_API vsi_bool vsi_nn_op_common_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

/**
 * Setup for eltwise binary op
 * Setup for eltwise binary op, which need broadcast shape.
 *
 * @param[in] node Node handle.
 * @param[in] inputs input tensors.
 * @param[in] outputs output tensors.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
OVXLIB_API vsi_bool vsi_nn_op_eltwise_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

OVXLIB_API vsi_status vsi_nn_op_common_optimize
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

vsi_bool vsi_nn_OpIsValid
    (
    vsi_nn_op_t op
    );

const vsi_nn_op_proc_t * vsi_nn_OpGetProc
    (
    vsi_nn_op_t op
    );

/**
 * Init operation
 * Call operation init process.
 *
 * @param[in] op Operation id.
 * @param[in] node Node handle.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
vsi_status vsi_nn_OpInit
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node
    );

/**
 * Build operation with vx backend
 * Call operation compute process, it will build the node with vx backend.
 *
 * @param[in] op Operation id.
 * @param[in] node Node handle.
 * @param[in] inputs Input tensors' handle..
 * @param[in] outputs Output tensors' handle.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
vsi_status vsi_nn_OpCompute
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

/**
 * Deinit operation
 * Call operation deinit process, free some resource.
 *
 * @param[in] op Operation id.
 * @param[in] node Node handle.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
vsi_status vsi_nn_OpDeinit
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node
    );

/**
 * Optimize operation
 * Call operation optimize process.
 * @see vsi_nn_opt_direction_e
 *
 * @param[in] op Operation id.
 * @param[in] node Node handle.
 * @param[in] inputs Input tensors' handle.
 * @param[in] outputs Output tensors' handle.
 * @param[in] driection Current loop direction, use this param to implement
 *                      different optimizations.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
vsi_status vsi_nn_OpOptimize
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    );

/**
 * Validate operation
 * Call operation check process.
 *
 * @param[in] op Operation id.
 * @param[in] node Node handle.
 * @param[in] inputs Input tensors' handle.
 * @param[in] outputs Output tensors' handle.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
vsi_bool vsi_nn_OpCheck
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

void vsi_nn_OpGetIoNum
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_size_t     * input_num,
    vsi_size_t     * output_num
    );

OVXLIB_API vsi_bool vsi_nn_OpGenerateTensor
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

/**
 * Setup operation
 * Call operation setup process, it runs before computation,
 * Ovxlib usually computes output shapes in this process.
 *
 * @param[in] op Operation id.
 * @param[in] node Node handle.
 * @param[in] inputs Input tensors' handle.
 * @param[in] outputs Output tensors' handle.
 *
 * @return VSI_SUCCESS on success, or error code otherwise.
 */
vsi_bool vsi_nn_OpSetup
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    );

vsi_bool vsi_nn_OpRegisterOvxInit
    (
    vsi_nn_op_t op,
    vsi_nn_op_compute_t compute
    );

vsi_bool vsi_nn_OpRegisterExternalOvxInit
    (
    vsi_nn_op_t op,
    const char* kernel_name,
    vsi_nn_op_proc_t* proc
    );

/**
 * Get operation name
 * Get operation name string by operation id.
 *
 * @param[in] op Operation id.
 *
 * @return Operation name on success, or NULL otherwise.
 */
OVXLIB_API const char * vsi_nn_OpGetName
    (
    vsi_nn_op_t op
    );

#if defined(__cplusplus)
}
#endif

/**
 * Declare an operation with process functions.
 */
#define DEF_OP_REG(op,init,compute,deinit,check,setup,optimize,in,out) \
    vsi_nn_op_proc_t vsi_nn_op_##op =\
{\
    /* init       */ init,\
    /* compute    */ compute,\
    /* deinit     */ deinit,\
    /* check      */ check,\
    /* setup      */ setup,\
    /* optimize   */ optimize,\
    /* input_num  */ in,\
    /* output_num */ out \
};

#endif
