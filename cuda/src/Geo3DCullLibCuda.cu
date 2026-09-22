// ============================================================================
//  Geo3DCullLibCuda.cu - CUDA host wrappers
// ----------------------------------------------------------------------------
//  Mirrors the CPU (OpenMP) API, implemented with one thread per vertex /
//  face on the GPU.  All CUDA runtime calls are checked with CUDA_CHECK (see
//  pch.h) so failures abort with a readable message instead of silently
//  corrupting the result arrays.
// ============================================================================
#include "pch.h"
#include "Geo3DCullLibCuda.h"
#include "Geo3DCullLibCudaUtils.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// ----------------------------------------------------------------------------
//  testCuda - sanity check: enumerate devices
// ----------------------------------------------------------------------------
extern "C" GEO3DCULLLIB_CUDA_API
void testCuda()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cout << "No CUDA devices found!" << endl;
        return;
    }

    cout << "Found " << deviceCount << " CUDA device(s)" << endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        cout << "  Device " << i << ": " << prop.name << endl;
    }

    // Exercise the runtime by allocating a small buffer and a single-element
    // readback; this verifies the CUDA context is usable.
    int *d_probe = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_probe, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_probe, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    int probe = -1;
    CUDA_CHECK(cudaMemcpy(&probe, d_probe, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_probe));
    (void)probe;
}

extern "C" GEO3DCULLLIB_CUDA_API
void cuda_getVtsCullCamLoS(
    double* vts,
    double* vns,
    int* V,
    int* fcs,
    double* fns,
    double* vfs,
    int* F,
    double* camPos,
    double* camDir,
    double* THR,
    bool* isVis)
{
    // Allocate device memory
    double *d_vts, *d_vns, *d_fns, *d_vfs, *d_camPos, *d_camDir;
    int *d_fcs;
    int *d_F, *d_V;
    double *d_THR;
    bool *d_isVis;

    size_t v_size = V[0] * 3 * sizeof(double);
    size_t f_size = F[0] * 3 * sizeof(int);
    size_t fns_size = F[0] * 3 * sizeof(double);
    size_t vfs_size = F[0] * 3 * sizeof(double);
    size_t camPos_size = 3 * sizeof(double);
    size_t camDir_size = 3 * sizeof(double);
    size_t thr_size = sizeof(double);
    size_t isVis_size = V[0] * sizeof(bool);

    CUDA_CHECK(cudaMalloc(&d_vts, v_size));
    CUDA_CHECK(cudaMalloc(&d_vns, v_size));
    CUDA_CHECK(cudaMalloc(&d_fcs, f_size));
    CUDA_CHECK(cudaMalloc(&d_fns, fns_size));
    CUDA_CHECK(cudaMalloc(&d_vfs, vfs_size));
    CUDA_CHECK(cudaMalloc(&d_camPos, camPos_size));
    CUDA_CHECK(cudaMalloc(&d_camDir, camDir_size));
    CUDA_CHECK(cudaMalloc(&d_THR, thr_size));
    CUDA_CHECK(cudaMalloc(&d_V, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_F, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_isVis, isVis_size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_vts, vts, v_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vns, vns, v_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fcs, fcs, f_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fns, fns, fns_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vfs, vfs, vfs_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_camPos, camPos, camPos_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_camDir, camDir, camDir_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_THR, THR, thr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, F, sizeof(int), cudaMemcpyHostToDevice));

    // Initialize isVis array on device
    CUDA_CHECK(cudaMemset(d_isVis, 1, isVis_size)); // Initialize to true

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (V[0] + threadsPerBlock - 1) / threadsPerBlock;

    getVtsCullCamLoSKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_vts, d_vns, V[0], d_fcs, d_fns, d_vfs, F[0],
        d_camPos, d_camDir, THR[0], d_isVis);

    CUDA_CHECK(cudaGetLastError());
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(isVis, d_isVis, isVis_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_vts));
    CUDA_CHECK(cudaFree(d_vns));
    CUDA_CHECK(cudaFree(d_fcs));
    CUDA_CHECK(cudaFree(d_fns));
    CUDA_CHECK(cudaFree(d_vfs));
    CUDA_CHECK(cudaFree(d_camPos));
    CUDA_CHECK(cudaFree(d_camDir));
    CUDA_CHECK(cudaFree(d_THR));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_isVis));
}

extern "C" GEO3DCULLLIB_CUDA_API
void cuda_getVisFcsVtsFromCull(
    int* fcs,
    int* F,
    bool* isVis,
    int* V,
    int* nTHR,
    bool* isFvis,
    bool* isVvis)
{
    // Allocate device memory
    int *d_fcs;
    int *d_F, *d_V, *d_nTHR;
    bool *d_isVis, *d_isFvis, *d_isVvis;

    size_t f_size = F[0] * 3 * sizeof(int);
    size_t isVis_size = V[0] * sizeof(bool);
    size_t isFvis_size = F[0] * sizeof(bool);
    size_t isVvis_size = V[0] * sizeof(bool);

    CUDA_CHECK(cudaMalloc(&d_fcs, f_size));
    CUDA_CHECK(cudaMalloc(&d_F, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_V, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nTHR, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_isVis, isVis_size));
    CUDA_CHECK(cudaMalloc(&d_isFvis, isFvis_size));
    CUDA_CHECK(cudaMalloc(&d_isVvis, isVvis_size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_fcs, fcs, f_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, F, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nTHR, nTHR, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_isVis, isVis, isVis_size, cudaMemcpyHostToDevice));

    // Initialize output arrays to "invisible"; the kernel sets them to true
    // where a face / vertex qualifies as visible.
    CUDA_CHECK(cudaMemset(d_isFvis, 0, isFvis_size));
    CUDA_CHECK(cudaMemset(d_isVvis, 0, isVvis_size));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (F[0] + threadsPerBlock - 1) / threadsPerBlock;

    getVisFcsVtsFromCullKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_fcs, F[0], d_isVis, V[0], nTHR[0], d_isFvis, d_isVvis);

    CUDA_CHECK(cudaGetLastError());
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(isFvis, d_isFvis, isFvis_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(isVvis, d_isVvis, isVvis_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_fcs));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_nTHR));
    CUDA_CHECK(cudaFree(d_isVis));
    CUDA_CHECK(cudaFree(d_isFvis));
    CUDA_CHECK(cudaFree(d_isVvis));
}