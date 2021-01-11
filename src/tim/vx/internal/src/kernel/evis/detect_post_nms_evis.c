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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS
#if 0
/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_DETECT_POST_NMS,
} _internal_kernel_e;

#define _DETECT_POST_NMS_KERNEL_SOURCE      "detect_post_nms"
#define _DETECT_POST_NMS_KERNEL_NAME        CVIVANTE_NAMESPACE("evis.detect_post_nms")

// Add kernel hashtable here
#define DETECT_POST_NMS_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { DETECT_POST_NMS_HASH_KEY( IN_DTYPE, OUT_DTYPE ), _DETECT_POST_NMS_KERNEL_NAME, SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _detect_post_nms_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, _DETECT_POST_NMS_KERNEL_SOURCE ),
};


/*
 * Kernel params
 */
static vx_param_description_t _detect_post_nms_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _DETECT_POST_NMS_PARAM_NUM  _cnt_of_array( _detect_post_nms_kernel_param_def )

#define SCALAR_NMS_TYPE     (6)
#define SCALAR_MAX_NUM      (7)
#define SCALAR_MAX_CLASS    (8)
#define SCALAR_MAX_DETECT   (9)
#define SCALAR_SCORE_TH     (10)
#define SCALAR_IOU_TH       (11)
#define SCALAR_IS_BG        (12)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_detect_post_nms_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;

    return status;
} /* _detect_post_nms_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;

    return status;
} /* _query_kernel() */

#endif

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

    return NULL;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( detect_post_nms, _setup )

