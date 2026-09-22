#pragma once

// Global header for Geo3DCullLib
//
// Cross-platform DLL / shared-library export macro.
// On Windows we use __declspec; on Linux/macOS we rely on the compiler's
// default visibility (the symbol is exported as soon as it is not static),
// so the macro expands to nothing there.

#if defined(_WIN32) || defined(_WIN64)
    #ifdef GEO3DCULLLIB_EXPORTS
        #define GEO3DCULLLIB_API __declspec(dllexport)
    #else
        #define GEO3DCULLLIB_API __declspec(dllimport)
    #endif // GEO3DCULLLIB_EXPORTS
#else
    #if defined(__GNUC__) && __GNUC__ >= 4
        #define GEO3DCULLLIB_API __attribute__((visibility("default")))
    #else
        #define GEO3DCULLLIB_API
    #endif
#endif // _WIN32/_WIN64

extern "C" {
    GEO3DCULLLIB_API void testOpenMP(); // Testing OpenMP Enabled

    GEO3DCULLLIB_API void OMP_getVtsCullCamLoS(
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

    GEO3DCULLLIB_API void OMP_getVisFcsVtsFromCull(
        int* fcs,         // (Fx3 integer) faces triangulation of vertices
        int* F,           // (1x1 integer) cardinality of faces
        bool* isVis,      // (Nx1 bool) TEMPORARY boolean array indicating the visibility of each vertex in the line-of-sight of the camera. (OUTPUT of OMP_getVtsCullCamLoS)
        int* V,           // (1x1 integer) cardinality of the vertices
        int* nTHR,        // (1x1 integer) cardinality threshold of visible vertices per triangular face, to relax the visible boundary. It can either be [3, 2, 1] with 3 (MOST RESTRICTIVE), and 1 (MOST RELAXED) boundary.
        bool* isFvis,     // (Nx1 bool) FINAL boolean array indicating the visibility of each face of the triangulation in the line-of-sight of the camera (after boundary RELAXATION).
        bool* isVvis      // (Nx1 bool) FINAL boolean array indicating the visibility of each vertex of the 3D mesh in the line-of-sight of the camera (after boundary RELAXATION).
    );
}