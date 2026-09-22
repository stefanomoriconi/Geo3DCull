#pragma once

// CUDA header for Geo3DCullLib
#if defined(_WIN32) || defined(_WIN64)
    #ifdef GEO3DCULLLIB_CUDA_EXPORTS
        #define GEO3DCULLLIB_CUDA_API __declspec(dllexport)
    #else
        #define GEO3DCULLLIB_CUDA_API __declspec(dllimport)
    #endif // GEO3DCULLLIB_CUDA_EXPORTS
#else
    #define GEO3DCULLLIB_CUDA_API
#endif // _WIN32/_WIN64

extern "C" {
    GEO3DCULLLIB_CUDA_API void testCuda(); // Testing CUDA functionality
    
    GEO3DCULLLIB_CUDA_API void cuda_getVtsCullCamLoS(
        double* vts,      // (Vx3 double) vertices of the geometry in 3D
        double* vns,      // (Vx3 double) vertices normals in 3D
        int* V,           // (1x1 integer) cardinality of the vertices and normals
        int* fcs,         // (Fx3 integer) faces triangulation of vertices
        double* fns,      // (Fx3 double) faces normals in 3D
        double* vfs,      // (Fx3 double) coordinates of faces centres (as vertices) in 3D
        int* F,           // (1x1 integer) cardinality of faces
        double* camPos,   // (1x3 double) coordinates of the camera position in 3D
        double* camDir,   // (1x3 double) pointing direction of the camera in 3D
        double* THR,      // (1x1 double) threshold distance off to the vertex-camera line in order to consider neighbouring faces to test for occlusions.
        bool* isVis       // (Nx1 bool) TEMPORARY boolean array indicating the visibility of each vertex in the line-of-sight of the camera.
    );

    GEO3DCULLLIB_CUDA_API void cuda_getVisFcsVtsFromCull(
        int* fcs,         // (Fx3 integer) faces triangulation of vertices
        int* F,           // (1x1 integer) cardinality of faces
        bool* isVis,      // (Nx1 bool) TEMPORARY boolean array indicating the visibility of each vertex in the line-of-sight of the camera. (OUTPUT of cuda_getVtsCullCamLoS)
        int* V,           // (1x1 integer) cardinality of the vertices
        int* nTHR,        // (1x1 integer) cardinality threshold of visible vertices per triangular face, to relax the visible boundary. It can either be [3, 2, 1] with 3 (MOST RESTRICTIVE), and 1 (MOST RELAXED) boundary.
        bool* isFvis,     // (Nx1 bool) FINAL boolean array indicating the visibility of each face of the triangulation in the line-of-sight of the camera (after boundary RELAXATION).
        bool* isVvis      // (Nx1 bool) FINAL boolean array indicating the visibility of each vertex of the 3D mesh in the line-of-sight of the camera (after boundary RELAXATION).
    );
}