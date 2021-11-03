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
#ifndef _VSI_NN_DAEMON_H
#define _VSI_NN_DAEMON_H

#ifdef __cplusplus
    #define _INITIALIZER(f) \
        static void f(void); \
        struct f##_t_{ f##_t_(void) { f(); }}; static f##_t_ f##_; \
        static void f(void)

    #define _DEINITIALIZER(f) \
        static void f(void); \
        struct f##_t_{ ~f##_t_(void) { f(); }}; static f##_t_ f##_; \
        static void f(void)

#elif defined(_MSC_VER)
    #pragma section(".CRT$XCU", read)
    #define _INITIALIZER2(f, p) \
        static void f(void); \
        __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
        __pragma(comment(linker, "/include:" p #f "_")) \
        static void f(void)
    #ifdef _WIN64
        #define _INITIALIZER(f) _INITIALIZER2(f, "")
    #else
        #define _INITIALIZER(f) _INITIALIZER2(f, "_")
    #endif

    #define _DEINITIALIZER(f) \
        static void f(void)

#elif defined(__linux__)
    #define _INITIALIZER(f) \
        static void f(void) __attribute__((constructor)); \
        static void f(void)

    #define _DEINITIALIZER(f) \
        static void f(void) __attribute__((destructor)); \
        static void f(void)

#else
    #error: Unsupport compiler.
#endif
#endif
