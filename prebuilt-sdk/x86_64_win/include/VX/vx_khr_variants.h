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

#ifndef _VX_KHR_VARIANT_H_
#define _VX_KHR_VARIANT_H_

/*!
 * \file
 * \brief The Khronos Extension for Kernel Variants.
 *
 * \defgroup group_variants Extension: Kernel Variants
 * \brief The Khronos Extension for Kernel Variants.
 * \details Kernel Variants allow the Client-Defined Functions to create several
 * kernels on the same target with the same name, but with slight variations
 * between them. Frequently these variants are expected to employ different
 * algorithms or methodologies.
 *
 * All target specific kernels and target variants must conform to the same OpenVX
 * specification of the OpenVX Kernel in order to use the string name and enumeration.
 * For example, a vendor may supply multiple targets,
 * and implement the same functionality on each. Futhermore the same
 * vendor may offer a variant on some specific target which offers some differentiation but
 * still  conforms to the definition of the OpenVX Kernel.
 * In this example there are 3 implementations of the same computer vision function, "Sobel3x3".
 * \arg On "CPU" a "Sobel3x3" which is "faster". A variant which may produce slightly less accurate but still conformant results.
 * \arg On "CPU" a "Sobel3x3" which is more "accurate". A variant which may run slower but produces bit exact results.
 * \arg On "GPU" a "Sobel3x3" \e default variant which may run on a remote core and produce bit exact results.
 *
 * In each of the cases a client of OpenVX could request the kernels in nearly
 * the same the same manner. There are two main approaches, which depend on the
 * method a client calls to get the kernel reference. The first uses enumerations.
 * This method allows to client to attempt to find other targets and variants, but if
 * these are not present, the default node would still have been constructed.
 * The second method depends on using fully qualified strings to get the kernel reference.
 * This second method is more compact but is does not permit fail-safing to default versions.
 *
 * As part of this extension, the function <tt>vxGetKernelByName</tt> will now accept more
 * qualifications to the string naming scheme. Kernels names can be additionally
 * qualified in 2 separate ways, by target and by variant. A "fully" qualified name is in the format of
 * <i>target</i><b>:</b><i>kernel</i><b>:</b><i>variant</i>.
 * Both \e target and \e variant may be omitted (for an unqualified name).
 * In this case, the implementation will assume the "default" value of these
 * names (which could literally be "default"). Names may also be fully
 * qualified with target included.
 * Examples:
 * \arg "khronos.c_model:org.khonos.openvx.sobel3x3:default" - fully qualified
 * \arg "org.khronos.openvx.sobel3x3:default" (missing target) - partially qualified
 * \arg "khronos.c_model:org.khronos.openvx.sobel3x3" (missing variant) - partially qualifed.
 * \arg "org.khronos.openvx.sobel3x3" - unqualified.
 *
 */

/*! \brief The string name of the extension.
 * \ingroup group_variants
 */
#define OPENVX_KHR_VARIANTS  "vx_khr_variants"

/*! \brief Defines the maximum number of characters in a variant string.
 * \ingroup group_variants
 */
#define VX_MAX_VARIANT_NAME (64)

#include <VX/vx.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Used to choose a variant of a kernel for execution on a particular node.
 * \param [in] node The reference to the node.
 * \param [in] variantName The name of the variant to choose.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_variants
 */
VX_API_ENTRY vx_status VX_API_CALL vxChooseKernelVariant(vx_node node, vx_char variantName[VX_MAX_VARIANT_NAME]);

#ifdef __cplusplus
}
#endif

#endif

