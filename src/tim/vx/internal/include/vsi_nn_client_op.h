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
#ifndef _VSI_NN_OP_CLIENT_H
#define _VSI_NN_OP_CLIENT_H

/*------------------------------------
               Includes
  -----------------------------------*/

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_ops.h"

#if defined(__cplusplus)
extern "C"{
#endif

/*------------------------------------
                Types
  -----------------------------------*/

/*------------------------------------
                Macros
  -----------------------------------*/

#define VSI_NN_DEF_CLIENT_OPS( ops, idx )   ( VSI_NN_OP_##ops## = VSI_NN_OP_CLIENT + idx )

/*------------------------------------
              Functions
  -----------------------------------*/

OVXLIB_API vsi_bool vsi_nn_OpIsRegistered
    (
    vsi_nn_op_t op
    );

OVXLIB_API vsi_bool vsi_nn_OpRegisterClient
    (
    vsi_nn_op_t op,
    vsi_nn_op_proc_t * proc
    );

OVXLIB_API vsi_nn_op_proc_t * vsi_nn_OpGetClient
    (
    vsi_nn_op_t op
    );

OVXLIB_API void vsi_nn_OpRemoveClient
    (
    vsi_nn_op_t op
    );

vsi_bool vsi_nn_OpAddClientName
  (
    vsi_nn_op_t op,
    const char* kernel_name
  );

const char* vsi_nn_OpGetClientName
  (
    vsi_nn_op_t op
  );

#if defined(__cplusplus)
}
#endif

#endif
