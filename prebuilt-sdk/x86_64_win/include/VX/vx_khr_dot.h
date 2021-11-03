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

#ifndef _VX_KHR_DOT_H_
#define _VX_KHR_DOT_H_

#define OPENVX_KHR_DOT  "vx_khr_dot"

#include <VX/vx.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Exports a single graph to a dotfile.
 * \param [in] graph The graph to export.
 * \param [in] dotfile The name of the file to write to.
 * \param [in] showData If true, data objects will be listed in the graph too.
 * \see http://www.graphviz.com
 */
vx_status vxExportGraphToDot(vx_graph g, vx_char dotfile[], vx_bool showData);

#ifdef __cplusplus
}
#endif

#endif

