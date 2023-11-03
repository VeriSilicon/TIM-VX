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

#ifndef _VSI_NN_GRAPH_H
#define _VSI_NN_GRAPH_H

#include "vsi_nn_platform.h"
#include "vsi_nn_context.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_types.h"
#include "vsi_nn_rnn.h"
#include "utils/vsi_nn_map.h"

/**
 * Default max node input or output tensors' number.
 * This value may be changed if some node's IO transcent
 * it.
 * @see vsi_nn_AddNode
 * */
#define VSI_NN_MAX_IO_NUM        32

/**
 * Default preprocess and postprocess node base uid.
 * When add new preprocess node in
 * graph, node uid is set based on it.
 * @see vsi_nn_AddPreprocNode
 * */
#define VSI_NN_PREPROC_NODE_UID_BASE    10000

/**
 * Default postprocess node base uid.
 * When add new postprocess node in
 * graph, node uid is set based on it.
 * @see vsi_nn_AddPostprocNode
 * */
#define VSI_NN_POSTPROC_NODE_UID_BASE   20000

/**
 * Default data convert node base uid.
 * When add new data convert node in
 * graph, node uid is set based on it.
 * @see vsi_nn_AddPreprocNode
 * */
#define VSI_NN_DATACONVERT_NODE_UID_BASE    30000

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Graph structure
 */
struct VSI_PUBLIC_TYPE _vsi_nn_graph
{
    /** Context */
    vsi_nn_context_t   ctx;
    /** OpenVX graph */
    vx_graph           g;
    /** Tensor list of this graph */
    union
    {
    /** @deprecated Never use tensors. */
    vsi_nn_tensor_t ** tensors;
    /** Tensor table */
    vsi_nn_map_t     * tensor_table;
    };
    union
    {
        /** Current tensor id */
        uint32_t          cur_tid;
        /** Tensor number */
        uint32_t          tensor_num;
    };
    /** @deprecated Max tensor number */
    uint32_t          max_tensor_num;
    /** Node list of this graph */
    union
    {
    /** @deprecated: Never use nodes. */
    vsi_nn_node_t   ** nodes;
    /** Node table */
    vsi_nn_map_t     * node_table;
    };
    union
    {
        /** Current node id */
        uint32_t          cur_nid;
        /** Node number */
        uint32_t          node_num;
    };
    /** @deprecated Max node number  */
    uint32_t          max_node_num;
    /** Max node input or output number */
    uint32_t          max_node_io;
    /** Inputs to the graph */
    struct
    {
        vsi_nn_tensor_id_t * tensors;
        uint32_t            num;
    } input;

    /** Outputs to the graph */
    struct
    {
        vsi_nn_tensor_id_t * tensors;
        uint32_t            num;
    } output;

    /** workspace for RNN */
    void* rnn_wksp;

    /** Handle manager */
    vsi_nn_handle_manager_t handle_manager;

    /** graph version */
    struct
    {
        uint32_t major;
        uint32_t minor;
        uint32_t patch;
    } version;

    /** Complete signal */
    struct
    {
        /** Flag to indicate if the need to append complete singal. */
        vsi_bool exists;
        union
        {
        /** Value to be sent. */
        int64_t value;
        /** Reserve some more btyes for future features. */
        uint8_t _bytes[64];
        };
        /** Length is not used yet, currently it will be always 8 bytes. */
        int32_t length;
        /** COMPLETE signal write address. */
        void* write_address;
        /** Pointer that store complete signal tensor,
         * this will automatic created after graph setup,
         * so please keep it NULL.*/
        vsi_nn_tensor_t* tensor;
    } complete_signal;

    vsi_bool isAllowFastMode;

    //DO NOT modify this sturct.
};

/**
 * Create graph
 * Create a new graph.
 *
 * @param[in] ctx Context to handle the graph.
 * @param[in] tensor_num Number of tensors to be created,
 *        set it 0 if it is unknown.
 * @param[in] node_num Number of nodes to be created,
 *        set it 0 if it is unknown.
 *
 * @return Graph hanlde, or NULL if create fail.
 *
 */
OVXLIB_API vsi_nn_graph_t * vsi_nn_CreateGraph
    (
    vsi_nn_context_t ctx,
    uint32_t        tensor_num,
    uint32_t        node_num
    );

/**
 * Release graph
 * Relase graph and set graph handle to NULL.
 *
 * @param[in] graph Graph handle pointer to release.
 *
 */
OVXLIB_API void vsi_nn_ReleaseGraph
    (
    vsi_nn_graph_t ** graph
    );

/**
 * Setup graph
 * Build graph with openVX tensors and nodes.
 *
 * @param[in] graph Graph handle
 * @param[in] sort If need to sort nodes.
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_SetupGraph
    (
    vsi_nn_graph_t * graph,
    vsi_bool          sort
    );

/**
 * Verify graph
 * Verify graph, this must be called after graph setup.
 *
 * @param[in] graph Graph handle
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 * */
OVXLIB_API vsi_status vsi_nn_VerifyGraph
    (
    vsi_nn_graph_t * graph
    );

/**
 * Run graph
 * Invoke the all nodes in graph.
 *
 * @param[in] graph Graph handle
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_RunGraph
    (
    vsi_nn_graph_t * graph
    );

/**
 * Genearate NBG cache
 * Genearate NBG cache
 *
 * @param[in] graph Graph handle
 * @param[in] nbg buffer pointer
 * @param[in] nbg buffer size
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_GenerateNBG(
    vsi_nn_graph_t * graph,
    void * nbg_buffer,
    size_t * size
    );

/**
 * Run graph in asynch way
 * Invoke the all nodes in graph.
 *
 * @param[in] graph Graph handle
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_AsyncRunGraph
    (
    vsi_nn_graph_t * graph
    );

OVXLIB_API vsi_status vsi_nn_AsyncRunWait
    (
    vsi_nn_graph_t * graph
    );

/**
 * Set graph version
 * Set the specific ovxlib version, this is used to fetch the
 * implementations of different ovxlib.
 *
 * @param[in] graph Graph handle
 * @param[in] major Ovxlib major version bined to graph
 * @param[in] minor Ovxlib minor version bined to graph
 * @param[in] patch Ovxlib patch version bined to graph
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_SetGraphVersion
    (
    vsi_nn_graph_t * graph,
    uint32_t major,
    uint32_t minor,
    uint32_t patch
    );

/**
 * Get graph version
 * Get Ovxlib version binded to graph.
 *
 * @param[in] graph Graph handle
 * @param[out] major Ovxlib major version binded to graph.
 * @param[out] minor Ovxlib minor version binded to graph.
 * @param[out] patch Ovxlib patch version binded to graph.
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_GetGraphVersion
    (
    vsi_nn_graph_t * graph,
    uint32_t *major,
    uint32_t *minor,
    uint32_t *patch
    );

/**
 * Add a new tensor to graph
 * Create a new tensor and add it to graph.
 *
 * @param[in] graph Graph handle
 * @param[in] id Optional id to the tensor, set it to VSI_NN_TENSOR_ID_AUTO,
 *           and a new id will be generated.
 * @param[in] attr Tensor attirbutes to the new tensor.
 * @param[in] data Optional data to the new tensor, if it's not NULL,
 *             the mem will be copied to the tensor mem.
 *
 * @return The new tensor id on success, or VSI_NN_TENSOR_ID_NA otheriwse.
 */
OVXLIB_API vsi_nn_tensor_id_t vsi_nn_AddTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr,
    /* Optional */
    uint8_t             * data
    );

/**
 * Add a new tensor from handle
 * Create a new tensor from a mem handle and add it to graph.
 *
 * @param[in] graph Graph handle
 * @param[in] id Optional id to the tensor, set it to VSI_NN_TENSOR_ID_AUTO,
 *           and a new id will be generated.
 * @param[in] attr Tensor attirbutes to the new tensor.
 * @param[in] data Optional mem handle to the new tensor, the new
 *             tensor will use this mem as its own mem handle,
 *             the mem handle must be 64 bytes align.
 *             If it's set to NULL, a new 64 bytes align mem handle will
 *             be automatic malloc.
 *
 * @return The new tensor id on success, or VSI_NN_TENSOR_ID_NA otheriwse.
 */
OVXLIB_API vsi_nn_tensor_id_t vsi_nn_AddTensorFromHandle
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr,
    uint8_t             * data
    );

/**
 * Add a new tensor from view
 * Create a new tensor from a view and add it to graph.
 *
 * @param[in] graph Graph handle.
 * @param[in] id Required, the id of the parent tensor on which to create view.
 * @param[in] start The start cooridinates for each dim, 0-based none-negative interger.
 *             NULL means copy from the idx 0 of each dim.
 * @param[in] end The end cooridinates for each dim, 0-based none-negative interger.
 *             NULL means copy to the end of each dim. For the given idx, the end[idx]
 *             should be greater than start[idx].
 * @return The new tensor id on success, or VSI_NN_TENSOR_ID_NA otheriwse.
 */
OVXLIB_API vsi_nn_tensor_id_t vsi_nn_AddTensorFromView
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t id,
    vsi_size_t* start,
    vsi_size_t* end
    );

/**
 * Attach tensor to graph
 * Attach an exist tensor to graph.
 *
 * @param[in] graph Graph handle
 * @param[in] id Optional id to the tensor, set it to VSI_NN_TENSOR_ID_AUTO,
 *           and a new id will be generated.
 * @param[in] tensor Tensor attach to the graph.
 *
 * @return The new tensor id on success, or VSI_NN_TENSOR_ID_NA otherwise.
 */
vsi_nn_tensor_id_t vsi_nn_AttachTensorToGraph
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_t      * tensor
    );

/**
 * @deprecated
 * @see vsi_nn_RemoveTensor
 */
void vsi_nn_DeleteTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id
    );

/**
 * Get tensor
 * Get tensor from graph.
 *
 * @param[in] graph Graph handle
 * @param[in] tensor_id Tensor's id
 *
 * @return Tensor's handle on success, or NULL otherwise.
 */
OVXLIB_API vsi_nn_tensor_t * vsi_nn_GetTensor
    (
    const vsi_nn_graph_t   * graph,
    vsi_nn_tensor_id_t tensor_id
    );

/**
 * Get node
 * Get node from graph.
 *
 * @param[in] graph Graph handle
 * @param[in] id Node's id
 *
 * @return Node's handle on success, or NULL otherwise.
 */
OVXLIB_API vsi_nn_node_t * vsi_nn_GetNode
    (
    const vsi_nn_graph_t   * graph,
    vsi_nn_node_id_t   id
    );

/**
 * Get tensors
 * Get multi tensors from graph.
 *
 * @param[in] graph Graph handle
 * @param[in] tensors_id Tensors' id
 * @param[in] num Number of tensors
 * @param[out] tensors Tensor handles on success, or NULL otherwise.
 */
OVXLIB_API void vsi_nn_GetTensors
    (
    vsi_nn_graph_t     * graph,
    vsi_nn_tensor_id_t * tensors_id,
    uint32_t            num,
    vsi_nn_tensor_t   ** tensors
    );

/**
 * Add node
 * Create a new node and attach it to graph.
 *
 * @param[in] graph Graph handle
 * @param[in] op Node operation.
 * @param[in] input_num Number of inputs to this node.
 * @param[in] output_num Number of outputs to this node.
 * @param[out] node_id A handle to get the id of new node,
 *                  pass it to NULL to get nothing.
 *
 * @return The node handle on success, or NULL otherwise.
 */
OVXLIB_API vsi_nn_node_t * vsi_nn_AddNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    uint32_t              input_num,
    uint32_t              output_num,
    vsi_nn_node_id_t    * node_id
    );

/**
 * Add External node
 * Create a new External node and attach it to graph.
 *
 * @param[in] graph Graph handle
 * @param[in] op Node operation.
 * @param[in] vsi_nn_proc_t to this node.
 * @param[in] output_num Number of outputs to this node.
 * @param[in] kernel name.
 * @param[out] node_id A handle to get the id of new node,
 *                  pass it to NULL to get nothing.
 *
 * @return The node handle on success, or NULL otherwise.
 */
OVXLIB_API vsi_nn_node_t * vsi_nn_AddExternalNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    const void           * proc,
    vsi_nn_node_id_t    * node_id,
    const char          *kernel_name
    );

/**
 * @deprecated
 * @see vsi_nn_AddNode
 */
OVXLIB_API vsi_nn_node_t * vsi_nn_AppendNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    vsi_nn_node_id_t    * node_id
    );

/**
 * Set graph inputs
 * Set inputs to the graph
 *
 * @param[in] graph Graph handle
 * @param[in] tensors_id Input tensors id to the graph.
 * @param[in] tensor_num Input tensors number.
 *
 * @return TRUE on success, or FALSE otherwise.
 */
OVXLIB_API vsi_bool vsi_nn_SetGraphInputs
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_tensor_id_t  * tensors_id,
    uint32_t             tensor_num
    );

/**
 * Set graph outputs
 * Set outputs to the graph
 *
 * @param[in] graph Graph handle
 * @param[in] tensors_id Output tensors id to the graph.
 * @param[in] tensor_num Output tensors number.
 *
 * @return TRUE on success, or FALSE otherwise.
 */
OVXLIB_API vsi_bool vsi_nn_SetGraphOutputs
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_tensor_id_t  * tensors_id,
    uint32_t             tensor_num
    );

/**
 * Remove node
 * Remove a node from graph. Please NOTE that, to remove a node
 * will break the connections of the node, so it is only used
 * when release a graph.
 *
 * @param[in] graph Graph handle
 * @param[in] id Node id to be removed.
 */
OVXLIB_API void vsi_nn_RemoveNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_node_id_t      id
    );

/**
 * Sort graph node
 * Sort the nodes with the execution sequence.
 *
 * @param[in] graph Graph handle
 *
 * @return Sorted nodes id. The node id buffer is malloc internal,
 *         the need to release it by user.
 */
vsi_nn_node_id_t * vsi_nn_SortGraphNode
    (
    vsi_nn_graph_t * graph
    );

/**
 * Get Nodes by uids
 * Get number of nodes with uids.
 *
 * @param[in] graph Graph handle
 * @param[in] node_uids Uids of nodes.
 * @param[in] node_uids_size Number of nodes.
 * @param[out] nodes Buffer to return node handles.
 * @param[in] nodes_num Why we need this? Number of nodes,
 *                 it must be equal to node_uids_size.
 *
 * @return Number of return node.
 */
OVXLIB_API uint32_t vsi_nn_GetNodesByUids
    (
    vsi_nn_graph_t   * graph,
    uint32_t        * node_uids,
    uint32_t          node_uids_size,
    vsi_nn_node_id_t * nodes,
    uint32_t          nodes_num
    );

/**
 * Dump node outputs
 * Dump outputs of given nodes.
 *
 * @param[in] graph Graph handle
 * @param[in] path A path to directory, all results will dump into it,
 * @param[in] node_uids Uids of dump nodes.
 * @param[in] node_uids_size Number of dump nodes.
 * @param[in] force_fp32 TRUE if all results needs to be converted to float32.
 * @param[in] data_fmt Not implemented.
 */
OVXLIB_API void vsi_nn_DumpGraphNodeOutputs
    (
    vsi_nn_graph_t * graph,
    const char     * path,
    uint32_t      *  node_uids,
    uint32_t         node_uids_size,
    vsi_bool         force_fp32,
    vsi_nn_dim_fmt_e data_fmt
    );

/**
 * Dump node outputs
 * Dump outputs of given nodes.
 *
 * @param[in] graph Graph handle
 * @param[in] path A path to directory, all results will dump into it,
 * @param[in] prefix A prefix of dump nodes.
 * @param[in] node_uids Uids of dump nodes.
 * @param[in] node_uids_size Number of dump nodes.
 * @param[in] force_fp32 TRUE if all results needs to be converted to float32.
 * @param[in] data_fmt Not implemented.
 */
OVXLIB_API void vsi_nn_DumpGraphNodeOutputsEx
    (
    vsi_nn_graph_t * graph,
    const char     * path,
    const char     * prefix,
    uint32_t       * node_uids,
    uint32_t         node_uids_size,
    vsi_bool         force_fp32,
    vsi_nn_dim_fmt_e data_fmt
    );

/**
 * Print graph
 * Print basic info of a graph.
 *
 * @param[in] graph Graph handle
 */
OVXLIB_API void vsi_nn_PrintGraph
    (
    vsi_nn_graph_t * graph
    );

/**
 * Dump graph to json
 * Dump basic info of a graph to json
 *
 * @param[in] graph Graph handle
 */
OVXLIB_API void vsi_nn_DumpGraphToJson
    (
    vsi_nn_graph_t *graph
    );

/**
 * Setup RNN Connections
 *
 * @param[in] graph Graph handle
 * @param[in] connections connections of RNN
 * @param[in] connections_count Number of connections
 * @see vsi_nn_rnn_external_connection_t
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise
 */
OVXLIB_API vsi_status vsi_nn_SetupRNNConnections
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_rnn_external_connection_t* connections,
    uint32_t connections_count
    );

/**
 * Reset RNN Buffers
 * Reset RNN buffers in graph
 *
 * @param[in] graph Graph handle
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise.
 */
OVXLIB_API vsi_status vsi_nn_ResetRNNBuffers
    (
    vsi_nn_graph_t* graph
    );

/**
 * Has RNN
 * Check if graph is a RNN
 *
 * @param[in] graph Graph handle
 *
 * @return TRUE if graph has RNN, or FALSE if not.
 */
OVXLIB_API vsi_bool vsi_nn_HasRNN
    (
    const vsi_nn_graph_t* graph
    );

/**
 * Remove tensor
 * Remove tensor from graph.
 *
 * @param[in] graph Graph handle
 * @param[in] id Tensor id
 */
void vsi_nn_RemoveTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id
    );

OVXLIB_API vsi_status vsi_nn_TrySetupCompleteSignalNode
    (
    vsi_nn_graph_t* graph
    );

vsi_status vsi_nn_setup_binary_graph_inputs_outputs
    (
    vsi_nn_graph_t* graph
    );

void  vsi_nn_get_tensor_consumers
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t tensor_id,
    vsi_nn_node_t** nodes,
    uint32_t* count
    );

void vsi_nn_get_tensor_provider
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t tensor_id,
    vsi_nn_node_t** node
    );

OVXLIB_API vsi_status vsi_nn_SetGraphPreloadSize
    (
    vsi_nn_graph_t* graph,
    vsi_nn_graph_attr_preload_type_e attr,
    uint32_t size
    );

vsi_nn_tensor_id_t vsi_nn_get_tensor_id
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_tensor_t * tensor
    );

OVXLIB_API vsi_status vsi_nn_SetGraphPriority
    (
    vsi_nn_graph_t* graph,
    uint32_t priority
    );

OVXLIB_API vsi_status vsi_nn_SetGraphFastMode
    (
    vsi_nn_graph_t* graph,
    vsi_bool fastmode
    );

OVXLIB_API vsi_bool vsi_nn_IsGraphFastMode
    (
    const vsi_nn_graph_t* graph
    );

OVXLIB_API vsi_status vsi_nn_CopyTensorViaGraphs
    (
    vsi_nn_graph_t *src_graph,
    vsi_nn_tensor_id_t src_tensor_id,
    vsi_nn_graph_t *dst_graph,
    vsi_nn_tensor_id_t dst_tensor_id
    );

OVXLIB_API vsi_status vsi_nn_ExecuteGraphLoop
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t *max_iteration_tensor
    );

OVXLIB_API vsi_status vsi_nn_SetGraphTransformOption
    (
    vsi_nn_graph_t* graph,
    const char* ctrl_str,
    size_t size
    );

#ifdef __cplusplus
}
#endif

#endif
