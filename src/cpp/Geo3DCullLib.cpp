// Core functionality for Geo3DCullLib
#include "pch.h"
#include "Geo3DCullLib.h"
#include "Geo3DCullLibUtils.h"
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

void testOpenMP()
{
#pragma omp parallel
    {
        char strOut[30];
        // snprintf is standard C99 and portable across MSVC/GCC/Clang
        // (unlike the MSVC-specific sprintf_s).
        snprintf(strOut, sizeof(strOut), "Hello from thread: %d", omp_get_thread_num());
        // Lock-free-enough for a diagnostic: guard the stream with omp critical
        // so interleaved thread output does not corrupt the line.
#pragma omp critical(stdio_out)
        cout << strOut << endl;
    }
}

void OMP_getVtsCullCamLoS(
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
    const double tol = 0.001;
    // clock_t begin = clock();

    // Parallelise over VERTICES. Each thread only ever writes to its own
    // isVis[v] entry, so the loop is race-free without any critical sections.
    // (The previous version spawned a brand-new OpenMP team *inside* the
    //  per-vertex loop -- see the "NOT WORKING" note -- which is both very
    //  slow and unnecessary for correctness.)
#pragma omp parallel for schedule(dynamic, 8)
    for (int v = 0; v < V[0]; v++)
    {
        double vn[] = { vns[v * 3 + 0], vns[v * 3 + 1], vns[v * 3 + 2] }; // Retrieving the vertex normal
        isVis[v] = prjct3D(vn, camDir) <= 0.0; // If the vertex normal is opposite to the camera direction (Potentially Visible Point)

        if (isVis[v])
        {
            double vt[] = { vts[v * 3 + 0], vts[v * 3 + 1], vts[v * 3 + 2] };
            double nnL[3];
            getLineDir(camPos, vt, nnL); // Computing the line unit-vector (direction) connecting the vertex-camera
            double dlos[3]; // Defining the difference vector: vertex-camera
            sbt3(vt, camPos, dlos);
            double d_dlos = normL2(dlos); // Computing the Direct Line-of-Sight distance of vertex-camera (magnitude of difference vector)

            for (int f = 0; f < F[0]; f++)
            {
                double fn[] = { fns[f * 3 + 0], fns[f * 3 + 1], fns[f * 3 + 2] };
                if (prjct3D(fn, camDir) <= 0.0) // SKIP ALL INVISIBLE FACES (face normal in the same direction of the camDir)
                {
                    // Retrieving the face center coordinates
                    double vf[] = { vfs[f * 3 + 0], vfs[f * 3 + 1], vfs[f * 3 + 2] };
                    if (normL2_pt2line(vf, camPos, nnL) <= THR[0]) // if the Face is within the point-line distance threshold
                    {
                        // Retrieving the Triangle Points Coordinates
                        double t0[] = { vts[fcs[f * 3 + 0] * 3 + 0], vts[fcs[f * 3 + 0] * 3 + 1], vts[fcs[f * 3 + 0] * 3 + 2] };
                        double t1[] = { vts[fcs[f * 3 + 1] * 3 + 0], vts[fcs[f * 3 + 1] * 3 + 1], vts[fcs[f * 3 + 1] * 3 + 2] };
                        double t2[] = { vts[fcs[f * 3 + 2] * 3 + 0], vts[fcs[f * 3 + 2] * 3 + 1], vts[fcs[f * 3 + 2] * 3 + 2] };
                        double ipt[3]; // Initialising intersection point
                        sectLineWTri(camPos, vt, t0, t1, t2, ipt); // Intersecting the line (vertex-camera) with the face-triangle
                        if (isPtInTri(ipt, t0, t1, t2))
                        {
                            double ilos[3]; // Defining the difference vector: vertex-camera
                            sbt3(ipt, camPos, ilos);
                            double d_ilos = normL2(ilos); // Computing the Direct Line-of-Sight distance of vertex-camera (magnitude of difference vector)

                            if (d_ilos < d_dlos - tol)
                            {
                                isVis[v] = false; // Occluded: this vertex is behind a closer face
                            }
                        }
                    }
                }

            }

        }

    }
    /* PRINTING TIME PERFORMANCE (Hard-CODED)
    clock_t end = clock();
    float elapsT = (float)(end - begin) / CLOCKS_PER_SEC;
    cout << "[PARALLEL 3] Computational Time: " << elapsT << endl;
    */
}

void OMP_getVisFcsVtsFromCull(
    int* fcs, 
    int* F, 
    bool* isVis, 
    int* V, 
    int* nTHR, 
    bool* isFvis, 
    bool* isVvis)
{
    // clock_t begin = clock();
    if (nTHR[0] <= 0)
    {
        nTHR[0] = 1;
    }

    // Initialize output arrays
    for (int f = 0; f < F[0]; f++)
    {
        isFvis[f] = false;
    }
    
    for (int v = 0; v < V[0]; v++)
    {
        isVvis[v] = false;
    }

    // Process each face to determine visibility
    for (int f = 0; f < F[0]; f++)
    {
        int f0 = fcs[f * 3 + 0];
        int f1 = fcs[f * 3 + 1];
        int f2 = fcs[f * 3 + 2];

        // Check if this face has enough visible vertices
        bool isFaceVisible = countVisibleVtsPerFace(f0, f1, f2, isVis, nTHR[0], isVvis);
        
        if (isFaceVisible)
        {
            isFvis[f] = true;
        }
    }

    // clock_t end = clock();
    // float elapsT = (float)(end - begin) / CLOCKS_PER_SEC;
    // cout << "[PARALLEL 4] Computational Time: " << elapsT << endl;
}