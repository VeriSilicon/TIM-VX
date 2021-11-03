#ifndef __VSI_NN_DLFCN_H
#define __VSI_NN_DLFCN_H

#if defined(_WIN32)
#define  RTLD_LAZY   0
#define  RTLD_NOW    0

#define  RTLD_GLOBAL (1 << 1)
#define  RTLD_LOCAL  (1 << 2)

#define  RTLD_DEFAULT    ((void *)0)
#define  RTLD_NEXT       ((void *)-1)

#else
#include <dlfcn.h>
#endif

/**
 * Opend a shared library
 *
 * @param[in] Library path
 * @param[in] Opend mode.
 *
 * @return Library handle on success, or NULL otherwise.
 */
void* vsi_nn_dlopen
    (
    const char *file,
    int mode
    );

/**
 * Close the opened library
 *
 * @param[in] Library handler
 *
 * @return TRUE on success
 */
int vsi_nn_dlclose
    (
    void *handle
    );

/**
 * Find symbol from opened library
 *
 * @param[in] Library handler
 * @param[in] Symbol name to find.
 *
 * @return Symbol
 */
void* vsi_nn_dlsym
    (
    void *handle,
    const char *name
    );

/**
 * Get error info.
 *
 * @return Error message.
 */
char * vsi_nn_dlerror(void);
#endif

