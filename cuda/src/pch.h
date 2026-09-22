// ============================================================================
//  Geo3DCullLibCuda - pch.h
// ----------------------------------------------------------------------------
//  Precompiled / common header for the CUDA translation units
//  (Geo3DCullLibCuda.cu / Geo3DCullLibCudaUtils.cu).
//
//  It provides:
//    * the standard C++ headers used by the host wrappers, and
//    * a small CUDA error-checking helper (CUDA_CHECK) so that runtime API
//      failures abort with a readable message instead of silently corrupting
//      results.
// ============================================================================
#ifndef _PCH_CUDA_H_
#define _PCH_CUDA_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

// Report a CUDA runtime error (with file/line) and abort.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                         "CUDA error %s (code %d) at %s:%d\n",                 \
                         cudaGetErrorString(_err), (int)_err, __FILE__,        \
                         __LINE__);                                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#endif  // _PCH_CUDA_H_
