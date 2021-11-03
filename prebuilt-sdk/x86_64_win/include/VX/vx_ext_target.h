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

#ifndef _VX_EXT_TARGET_H_
#define _VX_EXT_TARGET_H_

#include <VX/vx.h>

/*! \file
 * \brief The OpenVX Target API Definition
 */

/*! \brief The extension name.
 * \ingroup group_target
 */
#define OPENVX_EXT_TARGET "vx_ext_target"

/*! \brief Defines the maximum number of characters in a target string.
 * \ingroup group_target
 */
#define VX_MAX_TARGET_NAME (64)

enum vx_ext_target_context_attribute_e {
    /*! \brief Used to query the context for the number of active targets. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_CONTEXT_TARGETS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_CONTEXT) + 0xE,
};

/*! \brief An abstract handle to a target.
 * \ingroup group_target
 */
typedef struct _vx_target *vx_target;

/*! \brief The target attributes list
 * \ingroup group_target
 */
enum vx_target_attribute_e {
    /*! \brief Returns the index of the given target. Use a <tt>\ref vx_uint32</tt> parameter.*/
    VX_TARGET_ATTRIBUTE_INDEX = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_TARGET) + 0x0,
    /*! \brief Returns the name of the given target in the format "vendor.vendor_string".
     * Use a <tt>\ref vx_char</tt>[<tt>\ref VX_MAX_TARGET_NAME</tt>] array
     */
    VX_TARGET_ATTRIBUTE_NAME = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_TARGET) + 0x1,
    /*! \brief Returns the number of kernels that the target is capable of processing.
     * This is then used to allocate a table which is then filled when <tt>\ref vxQueryTarget</tt>
     * is called with <tt>\ref VX_TARGET_ATTRIBUTE_KERNELTABLE</tt>.
     * Use a <tt>\ref vx_uint32</tt> parameter.
     */
    VX_TARGET_ATTRIBUTE_NUMKERNELS = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_TARGET) + 0x2,
    /*! \brief Returns the table of all the kernels that a given target can execute.
     *  Use a <tt>vx_kernel_info_t</tt> array.
     * \pre You must call <tt>\ref vxQueryTarget</tt> with <tt>\ref VX_TARGET_ATTRIBUTE_NUMKERNELS</tt>
     * to compute the necessary size of the array.
     */
    VX_TARGET_ATTRIBUTE_KERNELTABLE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_TARGET) + 0x3,
};

#if defined(__cplusplus)
extern "C" {
#endif

/*! \brief Used to retrieve a target reference by the index of the target.
 * \param [in] context The reference to the overall context.
 * \param [in] index The index of the target to get a reference to.
 * \return <tt>\ref vx_target</tt>
 * \retval 0 Invalid index.
 * \retval * A target reference.
 * \note Use <tt>\ref vxQueryContext</tt> with <tt>\ref VX_CONTEXT_NUMTARGETS</tt> to retrieve the upper limit of targets.
 * \ingroup group_target
 */
VX_API_ENTRY vx_target VX_API_CALL vxGetTargetByIndex(vx_context context, vx_uint32 index);

/*! \brief Used to get a reference to named target when the name is known beforehand.
 * \param [in] context The reference to the overall context.
 * \param [in] name The target string name.
 * \return <tt>\ref vx_target</tt>
 * \retval 0 Invalid index.
 * \retval * A target reference.
 * \ingroup group_target
 */
VX_API_ENTRY vx_target VX_API_CALL vxGetTargetByName(vx_context context, const vx_char *name);

/*! \brief Releases a reference to a target object.
 * The object may not be garbage collected until its total reference count is zero.
 * \param [in] target The pointer to the target to release.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If target is not a <tt>\ref vx_target</tt>.
 * \note After returning from this function the reference will be zeroed.
 * \ingroup group_target
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseTarget(vx_target *target);

/*! \brief Used to query the target about it's properties.
 * \param [in] target The reference to the target.
 * \param [in] attribute The <tt>\ref vx_target_attribute_e</tt> value to query for.
 * \param [out] ptr The location at which the resulting value will be stored.
 * \param [in] size The size of the container to which ptr points.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \pre <tt>\ref vxGetTargetByName</tt> or <tt>\ref vxGetTargetByIndex</tt>
 * \ingroup group_target
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryTarget(vx_target target, vx_enum attribute, void *ptr, vx_size size);

/*! \brief Used to assign target affinity to a node.
 * \note This assignment overrides implementation chosen behavior.
 * \param [in] node The node reference to assign affinity to.
 * \param [in] target The reference to the target to execute the Node on.
 * \pre <tt>\ref vxGetTargetByName</tt> or <tt>\ref vxGetTargetByIndex</tt>
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_target
 * \pre <tt>vxCreateGenericNode</tt> or some other node creation function.
 * \retval VX_ERROR_INVALID_REFERENCE Either node or target was not a valid reference.
 * \retval VX_ERROR_NOT_SUPPORTED The node can not be executed on that target.
 */
VX_API_ENTRY vx_status VX_API_CALL vxAssignNodeAffinity(vx_node node, vx_target target);

#if defined(__cplusplus)
}
#endif

#endif
