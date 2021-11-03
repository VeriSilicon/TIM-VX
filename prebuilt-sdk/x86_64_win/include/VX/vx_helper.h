/* 

 * Copyright (c) 2012-2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _VX_HELPER_H_
#define _VX_HELPER_H_

#include <VX/vx.h>

/*! \file
 * \brief The OpenVX Helper Library Interface.
 *
 * \defgroup group_helper OpenVX Helper
 * \brief The helper is an non-standardized set of convenience constructs for OpenVX.
 * \details These functions use only the OpenVX API in order to implement their
 * functionality. As such structures, objects, defines, typedefs and functions
 * defined herein are not part of the OpenVX standard, and are
 * included as EXAMPLE code only.
 */

/*! \brief A definition for TAU, or 2*PI.
 * \ingroup group_helper
 */
#define VX_TAU 6.28318530717958647692

/*! \brief Maximum number of supported entries.
 * \ingroup group_helper
 */
#ifndef VX_MAX_LOG_NUM_ENTRIES
#define VX_MAX_LOG_NUM_ENTRIES (1024)
#endif

#ifndef dimof
/*! \brief A helper macro to determine the number of elements in an array.
 * \ingroup group_helper
 */
#define dimof(x)    (sizeof(x)/sizeof(x[0]))
#endif

/*! \brief Contains everything needed to abstractly describe a parameter to a kernel. This is used to
 * declare kernel parameters at compile time.
 * \ingroup group_helper
 */
typedef struct _vx_param_description_t {
    vx_enum     direction;      /*!< \brief From \ref vx_direction_e */
    vx_enum     data_type;      /*!< \brief From \ref vx_type_e */
    vx_enum     state;          /*!< \brief From \ref vx_parameter_state_e */
} vx_param_description_t;


/*! \brief Contains everything needed to abstractly describe a kernel.
 * This is used to declare kernels at compile time.
 * \ingroup group_helper
 */
typedef struct _vx_kernel_description_t {
    /*! \brief The vx_kernel_e enum */
    vx_enum                 enumeration;
    /*! \brief The name that kernel will be used with \ref vxGetKernelByName. */
    vx_char                 name[VX_MAX_KERNEL_NAME];
    /*! \brief The pointer to the function to execute the kernel */
    vx_kernel_f             function;
    /*! \brief The pointer to the array of parameter descriptors */
    vx_param_description_t *parameters;
    /*! \brief The number of paraemeters in the array. */
    vx_uint32               numParams;
    /*! \brief The parameters validator */
    vx_kernel_validate_f    validate;
    /*! \brief The input validator (deprecated  in openvx 1.1) */
    void* input_validate;
    /*! \brief The output validator (deprecated in openvx 1.1) */
    void* output_validate;
    /*! \brief The initialization function */
    vx_kernel_initialize_f initialize;
    /*! \brief The deinitialization function */
    vx_kernel_deinitialize_f deinitialize;
} vx_kernel_description_t;

/*! \brief A log entry contains the graph reference, a status and a message.
 * \ingroup group_helper
 */
typedef struct _vx_log_entry_t {
    /*! \brief The status code */
    vx_status    status;
    /*! \brief The reference to which the message and status pertains. */
    vx_reference reference;
    /*! \brief This indicates if the log entry is valid/active or not. */
    vx_enum      active;
    /*! \brief The message given to the log from OpenVX. This may be an empty string. */
    char         message[VX_MAX_LOG_MESSAGE_LEN];
} vx_log_entry_t;

/*! \brief The log of a graph
 * \ingroup group_helper
 */
typedef struct _vx_log_t {
    vx_int32  first;        /*!< Inclusive */
    vx_int32  last;         /*!< Exclusive */
    vx_uint32 count;        /*!< == VX_MAX_LOG_NUM_ENTRIES */
    /*! \brief The set of all log entries. */
    vx_log_entry_t entries[VX_MAX_LOG_NUM_ENTRIES];
} vx_log_t;

#define FGETS(str, fh)                              \
{                                                   \
    char* success = fgets(str, sizeof(str), fh);    \
    if (!success)                                   \
    {                                               \
        printf("fgets failed\n");                   \
    }                                               \
}

#ifdef __cplusplus
extern "C" {
#endif

uint32_t math_gcd(uint32_t a, uint32_t b);


/*! \brief Returns the previous entry of the log. When called consecutively it
 * will return the entire log. The log will be cleared by reading it.
 * \param [in] ref The reference to filter the log entries against.
 * If the context is given, the next entry will be returned.
 * \param [out] message A predefined location to store a copy of the log's
 * message value.
 * This must point to at least <tt>\ref VX_MAX_LOG_MESSAGE_LEN</tt> bytes of characters.
 * \return Returns the status of the log entry from <tt>\ref vx_status_e</tt>.
 * \ingroup group_helper
 * \note The API returns errors oldest to newest order.
 * When VX_SUCCESS is returned, the log reading is complete.
 */
vx_status vxGetLogEntry(vx_reference ref, char message[VX_MAX_LOG_MESSAGE_LEN]);

/*! \brief This enables the helper library logging feature to take over the error
 * log callback and keep a database of previous log entries.
 * \ingroup group_helper
 */
void vxRegisterHelperAsLogReader(vx_context context);

/*!
 * \brief A method to construct a node via arbitrary parameters and an enum.
 * \param [in] graph The handle to desired graph to add the node to.
 * \param [in] kernelenum The \ref vx_kernel_e enum value used to create a node.
 * \param [in] params The array of parameter information.
 * \param [in] num The number of elements in params.
 * \return vx_node
 * \retval 0 Indicates a failure.
 * \ingroup group_helper
 */
vx_node vxCreateNodeByStructure(vx_graph graph,
                                vx_enum kernelenum,
                                vx_reference params[],
                                vx_uint32 num);

/*! \brief A method to clear out the log for a particular reference, such as a graph.
 * \param [in] ref The reference to remove from the log.
 * \ingroup group_helper
 */
void vxClearLog(vx_reference ref);

/*! \brief This is used to connect one node parameter to another node parameter
 * when the original handles to the data objects are already lost.
 * The context determines if a buffer is necessary or can be optimized out.
 * \param [in] a The first parameter
 * \param [in] b The second parameter
 * \note a or b must be an output parameter and other other an input.
 * \return Returns a status code.
 * \ingroup group_helper
 */
vx_status vxLinkParametersByReference(vx_parameter a, vx_parameter b);

/*! \brief This is used to connect one parameter to another parameter by
 * explicity indexing when the handles to the data objects are lost.
 * \param [in] node_a The source node to link from.
 * \param [in] index_a The index of the \ref vx_parameter to link from.
 * \param [in] node_b The sink node to link to.
 * \param [in] index_b The index of the \ref vx_parameter to link to.
 * \return Returns a status code.
 * \ingroup group_helper
 */
vx_status vxLinkParametersByIndex(vx_node node_a, vx_uint32 index_a, vx_node node_b, vx_uint32 index_b);

/*! \brief This helper is used to easily set the affine matrix to a rotation and scale.
 * \param [in] matrix The handle to the matrix.
 * \param [in] angle The rotation angle in degrees.
 * \param [in] scale The scaling value. Values less than one are enlarging.
 * \param [in] center_x The center pixel in the x direction.
 * \param [in] center_y The center pixel in the y direction.
 * \return Returns a \ref vx_status_e enumeration.
 * \ingroup group_helper
 */
vx_status vxSetAffineRotationMatrix(vx_matrix matrix,
                                    vx_float32 angle,
                                    vx_float32 scale,
                                    vx_float32 center_x,
                                    vx_float32 center_y);

/*! \brief [Helper] This function changes the points of a rectangle by some
 * delta value per coordinate.
 * \param [in] rect The rectangle to modify.
 * \param [in] dsx The start x delta.
 * \param [in] dsy The start y delta.
 * \param [in] dex The end x delta.
 * \param [in] dey The end y delta.
 * \return vx_status
 * \retval VX_SUCCESS Modified rectangle.
 * \retval VX_ERROR_INVALID_REFERENCE Not a valid rectangle.
 * \ingroup group_helper
 */
vx_status vxAlterRectangle(vx_rectangle_t *rect,
                           vx_int32 dsx,
                           vx_int32 dsy,
                           vx_int32 dex,
                           vx_int32 dey);

/*! \brief Adds a parameter to a graph by indicating the source node, and the
 * index of the parameter on the node.
 * \param [in] g The graph handle.
 * \param [in] n The node handle.
 * \param [in] index The index of the parameter on the node.
 * \return Returns a \ref vx_status_e enumeration.
 * \ingroup group_helper
 */
vx_status vxAddParameterToGraphByIndex(vx_graph g, vx_node n, vx_uint32 index);

#if defined(EXPERIMENTAL_USE_TARGET)
/*! \brief Finds all targets which report that they implement a particular kernel by name.
 * \param [in] context The overall context.
 * \param [in] kname The name of the kernel to find.
 * \param [in,out] targets The array of pointers to character arrays. Each index will
 * be modified. If the kernel does not exist on the target, the name will be zeroed.
 * If the kernel does exist on the target, the name of the target will be filled in.
 * \pre targets must be a preallocated array of vx_char pointers to
 * <tt>\ref VX_MAX_TARGET_NAME</tt> characters with number of elements equal to
 * the number of targets in the implementation.
 * \ingroup group_helper
 */
vx_bool vxFindAllTargetsOfKernelsByName(vx_context context, vx_char kname[VX_MAX_KERNEL_NAME], vx_char *targets[]);

/*! \brief Allocates and returns a list of all available targets in a context.
 * \param [in] context The overall context.
 * \param [out] targets A pointer to variable to hold the array of target strings.
 * \param [out] num_targets A pointer to a variable to hold the number of targets found.
 * \ingroup group_helper
 */
vx_bool vxCreateListOfAllTargets(vx_context context, vx_char **targets[], vx_uint32 *num_targets);

/*! \brief Free the array of target name strings.
 * \param [in,out] targets The pointer to the variable that holds the array of strings. This variable will be set
 * to NULL after this call.
 * \param [in] num_targets The number of targets in the system.
 * \ingroup group_helper
 */
void vxDestroyListOfAllTargets(vx_char **targets[], vx_uint32 num_targets);

#endif

/*! \brief Find the overlapping rectange between two rectangles.
 * \ingroup group_helper
 */
vx_bool vxFindOverlapRectangle(vx_rectangle_t *rect_a, vx_rectangle_t *rect_b, vx_rectangle_t *rect_res);

/*! \brief Read a rectangle-shaped section of an image into a 2D array.
 * \ingroup group_helper
 */
void vxReadRectangle(const void *base,
                     const vx_imagepatch_addressing_t *addr,
                     const vx_border_t *borders,
                     vx_df_image type,
                     vx_uint32 center_x,
                     vx_uint32 center_y,
                     vx_uint32 radius_x,
                     vx_uint32 radius_y,
                     void *destination);

#ifdef __cplusplus
}
#endif

#endif  /* _VX_HELPER_H_ */

