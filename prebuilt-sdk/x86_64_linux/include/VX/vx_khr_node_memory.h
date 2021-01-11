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

#ifndef _VX_KHR_NODE_MEMORY_H_
#define _VX_KHR_NODE_MEMORY_H_

/*! \brief The Node Memory Extension.
 * \file
 */

#define OPENVX_KHR_NODE_MEMORY      "vx_khr_node_memory"

#include <VX/vx.h>

/*! \brief The kernel object attributes for global and local memory.
 * \ingroup group_kernel
 */
enum vx_kernel_attribute_memory_e {
    /*! \brief The global data pointer size to be shared across all instances of
     * the kernel (nodes are instances of kernels).
     * Use a \ref vx_size parameter.
     * \note If not set it will default to zero.
     */
    VX_KERNEL_GLOBAL_DATA_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x5,
    /*! \brief The global data pointer to the shared across all the instances of
     * the kernel (nodes are instances of the kernels).
     * Use a \ref void * parameter.
     */
    VX_KERNEL_GLOBAL_DATA_PTR = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x6,
};

/*! \brief The node object attributes for global and local memory.
 * \ingroup group_node
 */
enum vx_node_attribute_memory_e {
    /*! \brief Used to indicate the size of the shared kernel global memory area.
     * Use a \ref vx_size parameter.
     */
    VX_NODE_GLOBAL_DATA_SIZE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x9,
    /*! \brief Used to indicate the pointer to the shared kernel global memory area.
     * Use a void * parameter.
     */
    VX_NODE_GLOBAL_DATA_PTR = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0xA,
};

#endif

