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

#ifndef _VX_KHR_INTERP_H_
#define _VX_KHR_INTERP_H_

/*! \brief The Interpolation Type Query Extension.
 * \file
 */

#define OPENVX_KHR_INTERP   "vx_khr_interpolation"

#include <VX/vx.h>

/*! \brief Additional interpolation types */
enum vx_interpolation_type_ext_e {
    /*! \brief Bicubic interpolation method */
    VX_INTERPOLATION_BICUBIC = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_INTERPOLATION) + 0x3,
    /*! \brief Mipmapping interpolation method */
    VX_INTERPOLATION_MIPMAP = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_INTERPOLATION) + 0x4,
};

#endif

