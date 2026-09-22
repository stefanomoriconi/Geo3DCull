// ============================================================================
//  Geo3DCullLib -- End-to-end test program
// ----------------------------------------------------------------------------
//  Builds small test meshes in-code, calls the exported C API, and verifies
//  the expected visibility results.
//
//  Returns 0 on success, 1 on any failure.
//
//  Build & run (Linux, from refactored/):
//      make test
//
//  Build & run (Windows):
//      1. Build the DLL:  see Geo3DCullLib_refactored.vcxproj (Debug|x64)
//      2. Compile this file linking against x64/Debug/Geo3DCullLib.lib
//      3. Run test_program.exe from the repo root (DLL must be on PATH
//         or in the same directory).
// ============================================================================
#include <cstdio>
#include <cmath>
#include <cstring>
#include "Geo3DCullLib.h"

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s  (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } else { \
        std::printf("  OK:   %s\n", msg); \
    } \
} while(0)

// ============================================================================
//  Test 1 – Single triangle facing the camera (no occluder)
//  Triangle in xy-plane at z=0, normal +z.  Camera at (0.5,0.5,5) looking -z.
//  Expected: all 3 vertices visible, face visible at every nTHR.
// ============================================================================
static void testSingleVisibleTriangle() {
    std::printf("\n[Test 1] Single triangle facing camera (no occluder)\n");

    double vts[] = { 0,0,0,   1,0,0,   0,1,0 };
    double vns[] = { 0,0,1,   0,0,1,   0,0,1 };
    int V = 3;

    int    fcs[] = { 0, 1, 2 };
    double fns[] = { 0, 0, 1 };
    double vfs[] = { 1.0/3, 1.0/3, 0 };   // centroid
    int F = 1;

    double camPos[] = { 0.5, 0.5, 5.0 };
    double camDir[] = { 0.0, 0.0, -1.0 };
    double THR[]    = { 1.0 };

    bool isVis[3];
    std::memset(isVis, 0, sizeof(isVis));

    OMP_getVtsCullCamLoS(vts, vns, &V, fcs, fns, vfs, &F,
                         camPos, camDir, THR, isVis);

    CHECK(isVis[0], "vertex 0 visible");
    CHECK(isVis[1], "vertex 1 visible");
    CHECK(isVis[2], "vertex 2 visible");

    bool isFvis[1], isVvis[3];
    int  nTHR = 3;   // most restrictive
    OMP_getVisFcsVtsFromCull(fcs, &F, isVis, &V, &nTHR, isFvis, isVvis);

    CHECK(isFvis[0], "face visible (nTHR=3)");
    CHECK(isVvis[0] && isVvis[1] && isVvis[2],
          "all vertices visible after relaxation (nTHR=3)");
}

// ============================================================================
//  Test 2 – Larger front triangle occludes smaller back triangle
//
//   Both triangles share the centroid (0.5, 0.5) in the xy-plane. The front
//   triangle is a 2× scale-up of the back one, so every ray from the camera
//   to a BACK vertex crosses the z=1 plane strictly INSIDE the front triangle
//   (intersection lies within 0.24 of the incenter, inradius ≈ 0.30).
//
//   Back   (z=0):  (0.50,0.20) (0.76,0.65) (0.24,0.65)   normal +z
//   Front  (z=1):  (0.50,-0.10) (1.02,0.80) (-0.02,0.80) normal +z
//   Camera:        (0.5, 0.5, 5) looking -z  (on the symmetry axis)
//
//   For every FRONT vertex the ray's z=0 intersection falls BEHIND the vertex
//   (t=1.25 > t=1.0), so front vertices stay visible = nearest surface.
// ============================================================================
static void testOcclusion() {
    std::printf("\n[Test 2] Larger front triangle occluding smaller back triangle\n");

    double vts[] = {
        0.50,0.20,0.0,   0.76,0.65,0.0,   0.24,0.65,0.0,   // back  (z=0)
        0.50,-0.10,1.0,  1.02,0.80,1.0,  -0.02,0.80,1.0    // front (z=1)
    };
    double vns[] = {
        0,0,1,   0,0,1,   0,0,1,
        0,0,1,   0,0,1,   0,0,1
    };
    int V = 6;

    int    fcs[] = { 0,1,2,  3,4,5 };
    double fns[] = { 0,0,1,  0,0,1 };
    double vfs[] = { 0.50,0.50,0.0,   0.50,0.50,1.0 };
    int F = 2;

    double camPos[] = { 0.5, 0.5, 5.0 };
    double camDir[] = { 0.0, 0.0, -1.0 };
    double THR[]    = { 1.0 };

    bool isVis[6];
    std::memset(isVis, 0, sizeof(isVis));

    OMP_getVtsCullCamLoS(vts, vns, &V, fcs, fns, vfs, &F,
                         camPos, camDir, THR, isVis);

    // Back vertices should be occluded by the front triangle
    CHECK(!isVis[0], "back vertex 0 occluded");
    CHECK(!isVis[1], "back vertex 1 occluded");
    CHECK(!isVis[2], "back vertex 2 occluded");

    // Front vertices should be visible (nearest surface)
    CHECK( isVis[3], "front vertex 3 visible");
    CHECK( isVis[4], "front vertex 4 visible");
    CHECK( isVis[5], "front vertex 5 visible");

    // Face relaxation: back face NOT visible (0 of 3 verts visible < nTHR)
    bool isFvis[2], isVvis[6];
    int  nTHR = 3;
    OMP_getVisFcsVtsFromCull(fcs, &F, isVis, &V, &nTHR, isFvis, isVvis);

    CHECK(!isFvis[0], "back face NOT visible (nTHR=3)");
    CHECK( isFvis[1], "front face visible (nTHR=3)");
}

// ============================================================================
//  Test 3 – Backface culling
//  Triangle with normal (0,0,-1) facing AWAY from camera at +z.
//  prjct(vn, camDir) = (-1)(-1) = +1 > 0  →  vertex NOT visible.
// ============================================================================
static void testBackfaceCulling() {
    std::printf("\n[Test 3] Backface culling (normal facing away from camera)\n");

    double vts[] = { 0,0,0,   1,0,0,   0,1,0 };
    double vns[] = { 0,0,-1,  0,0,-1,  0,0,-1 };
    int V = 3;

    int    fcs[] = { 0, 1, 2 };
    double fns[] = { 0, 0, -1 };
    double vfs[] = { 1.0/3, 1.0/3, 0 };
    int F = 1;

    double camPos[] = { 0.5, 0.5, 5.0 };
    double camDir[] = { 0.0, 0.0, -1.0 };
    double THR[]    = { 1.0 };

    bool isVis[3];
    std::memset(isVis, 0, sizeof(isVis));

    OMP_getVtsCullCamLoS(vts, vns, &V, fcs, fns, vfs, &F,
                         camPos, camDir, THR, isVis);

    CHECK(!isVis[0], "vertex 0 NOT visible (backface)");
    CHECK(!isVis[1], "vertex 1 NOT visible (backface)");
    CHECK(!isVis[2], "vertex 2 NOT visible (backface)");

    bool isFvis[1], isVvis[3];
    int  nTHR = 1;   // even the most relaxed threshold
    OMP_getVisFcsVtsFromCull(fcs, &F, isVis, &V, &nTHR, isFvis, isVvis);

    CHECK(!isFvis[0], "face NOT visible (nTHR=1, all backface)");
}

// ============================================================================
int main() {
    std::printf("=== Geo3DCullLib Test Program ===\n");

    std::printf("\n[Setup] Verifying OpenMP support...\n");
    testOpenMP();

    testSingleVisibleTriangle();
    testOcclusion();
    testBackfaceCulling();

    std::printf("\n===================================\n");
    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}