// CUDA utility functions for Geo3DCullLib
#include "pch.h"
#include "Geo3DCullLibCudaUtils.h"
#include <math.h>

__device__ void getLineDirCuda(double p0L[3], double p1L[3], double nnL[3])
{
    double d0[3];
    sbt3Cuda(p1L, p0L, d0);
    uvect3DCuda(d0, nnL);
}

__device__ double normL2Cuda(double p0[3])
{
    return sqrt(pow(p0[0], 2.0) + pow(p0[1], 2.0) + pow(p0[2], 2));
}

__device__ double prjct3DCuda(double p0[3], double p1[3])
{
    return p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2];
}

__device__ void cross3DCuda(double u[3], double v[3], double w[3])
{
    w[0] = u[1] * v[2] - u[2] * v[1];
    w[1] = u[2] * v[0] - u[0] * v[2];
    w[2] = u[0] * v[1] - u[1] * v[0];
}

__device__ void uvect3DCuda(double p0[3], double n0[3])
{
    double L2 = normL2Cuda(p0);
    n0[0] = p0[0] / L2;
    n0[1] = p0[1] / L2;
    n0[2] = p0[2] / L2;
}

__device__ void add3Cuda(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] + v1[0];
    v2[1] = v0[1] + v1[1];
    v2[2] = v0[2] + v1[2];
}

__device__ void sbt3Cuda(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] - v1[0];
    v2[1] = v0[1] - v1[1];
    v2[2] = v0[2] - v1[2];
}

__device__ void mlt3Cuda(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] * v1[0];
    v2[1] = v0[1] * v1[1];
    v2[2] = v0[2] * v1[2];
}

__device__ void div3Cuda(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] / v1[0];
    v2[1] = v0[1] / v1[1];
    v2[2] = v0[2] / v1[2];
}

__device__ double normL2_pt2lineCuda(double pt[3], double ptL[3], double nnL[3])
{
    double ptD[3];
    sbt3Cuda(pt, ptL, ptD);
    double w[3];
    cross3DCuda(ptD, nnL, w);
    double d1 = normL2Cuda(w);
    double d2 = normL2Cuda(nnL);
    return d1 / d2;
}

__device__ void sectLineWTriCuda(double p0L[3], double p1L[3], double t0[3], double t1[3], double t2[3], double pLP[3])
{
    double segL[3];
    double negL[3];
    double d0[3];
    double d1[3];
    double w[3];
    double d2[3];
    sbt3Cuda(p1L, p0L, segL); //segment of the Line in 3D
    negL[0] = -1.0 * segL[0];
    negL[1] = -1.0 * segL[1];
    negL[2] = -1.0 * segL[2];
    sbt3Cuda(t1, t0, d0);
    sbt3Cuda(t2, t0, d1);
    cross3DCuda(d0, d1, w);
    sbt3Cuda(p0L, t0, d2);
    double t_N = prjct3DCuda(w, d2);
    double t_D = prjct3DCuda(negL, w);
    pLP[0] = p0L[0] + (segL[0] * (t_N / t_D));
    pLP[1] = p0L[1] + (segL[1] * (t_N / t_D));
    pLP[2] = p0L[2] + (segL[2] * (t_N / t_D));
}

__device__ bool isPtInTriCuda(double p0[3], double t0[3], double t1[3], double t2[3])
{
    double e0[3];
    double e1[3];
    double e2[3];

    sbt3Cuda(t2, t0, e0);
    sbt3Cuda(t1, t0, e1);
    sbt3Cuda(p0, t0, e2);

    double d00 = prjct3DCuda(e0, e0);
    double d01 = prjct3DCuda(e0, e1);
    double d02 = prjct3DCuda(e0, e2);
    double d11 = prjct3DCuda(e1, e1);
    double d12 = prjct3DCuda(e1, e2);

    double iD = 1.0 / ((d00 * d11) - (d01 * d01));
    double u = ((d11 * d02) - (d01 * d12)) * iD;
    double v = ((d00 * d12) - (d01 * d02)) * iD;

    return (u >= 0.0) && (v >= 0.0) && ((u + v) <= 1.0);
}

__device__ bool countVisibleVtsPerFaceCuda(int f0, int f1, int f2, bool* isVis, int nTHR, bool* isVvis)
{
    // Checking if the indexed vertex of the triangual face is Visible in the camera Line-of-Sight
    int counter = 0;
    if (isVis[f0])
        counter++;
    if (isVis[f1])
        counter++;
    if (isVis[f2])
        counter++;
    if (counter >= nTHR) // if the Triang
    {
        isVvis[f0] = true;
        isVvis[f1] = true;
        isVvis[f2] = true;
    }
    return counter >= nTHR;
}

__global__ void getVtsCullCamLoSKernel(
    double* vts, 
    double* vns, 
    int V,
    int* fcs, 
    double* fns, 
    double* vfs, 
    int F, 
    double* camPos, 
    double* camDir, 
    double THR, 
    bool* isVis)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v >= V) return;
    
    // Initialize visibility
    isVis[v] = true;
    
    const double tol = 0.001;
    
    double vn[3] = { vns[v * 3 + 0], vns[v * 3 + 1], vns[v * 3 + 2] }; // Retrieving the vertex normal
    isVis[v] = prjct3DCuda(vn, camDir) <= 0.0; // If the vertex normal is opposite to the camera direction (Potentially Visible Point)

    if (isVis[v])
    {
        double vt[3] = { vts[v * 3 + 0], vts[v * 3 + 1], vts[v * 3 + 2] };
        double nnL[3];
        getLineDirCuda(camPos, vt, nnL); // Computing the line unit-vector (direction) connecting the vertex-camera
        double dlos[3]; // Defining the difference vector: vertex-camera
        sbt3Cuda(vt, camPos, dlos);
        double d_dlos = normL2Cuda(dlos); // Computing the Direct Line-of-Sight distance of vertex-camera (magnitude of difference vector)

        // Process all faces for this vertex
        for (int f = 0; f < F; f++)
        {
            double fn[3] = { fns[f * 3 + 0], fns[f * 3 + 1], fns[f * 3 + 2] };
            if (prjct3DCuda(fn, camDir) <= 0.0) // SKIP ALL INVISIBLE FACES (face normal in the same direction of the camDir)
            {
                // Retrieving the face center coordinates
                double vf[3] = { vfs[f * 3 + 0], vfs[f * 3 + 1], vfs[f * 3 + 2] };
                if (normL2_pt2lineCuda(vf, camPos, nnL) <= THR) // if the Face is within the point-line distance threshold
                {
                    // Retrieving the Triangle Points Coordinates
                    double t0[3] = { vts[fcs[f * 3 + 0] * 3 + 0], vts[fcs[f * 3 + 0] * 3 + 1], vts[fcs[f * 3 + 0] * 3 + 2] };
                    double t1[3] = { vts[fcs[f * 3 + 1] * 3 + 0], vts[fcs[f * 3 + 1] * 3 + 1], vts[fcs[f * 3 + 1] * 3 + 2] };
                    double t2[3] = { vts[fcs[f * 3 + 2] * 3 + 0], vts[fcs[f * 3 + 2] * 3 + 1], vts[fcs[f * 3 + 2] * 3 + 2] };
                    double ipt[3]; // Initialising intersection point
                    sectLineWTriCuda(camPos, vt, t0, t1, t2, ipt); // Intersecting the line (vertex-camera) with the face-triangle
                    if (isPtInTriCuda(ipt, t0, t1, t2))
                    {
                        double ilos[3]; // Defining the difference vector: vertex-camera
                        sbt3Cuda(ipt, camPos, ilos);
                        double d_ilos = normL2Cuda(ilos); // Computing the Direct Line-of-Sight distance of vertex-camera (magnitude of difference vector)

                        if (d_ilos < d_dlos - tol)
                        {
                            isVis[v] = false;
                            return; // Early exit as visibility is now false
                        }
                    }
                }
            }
        }
    }
}

__global__ void getVisFcsVtsFromCullKernel(
    int* fcs, 
    int F, 
    bool* isVis, 
    int V, 
    int nTHR, 
    bool* isFvis, 
    bool* isVvis)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (f >= F) return;
    
    // Initialize face visibility
    isFvis[f] = false;
    
    // Get indices for this face
    int f0 = fcs[f * 3 + 0];
    int f1 = fcs[f * 3 + 1];
    int f2 = fcs[f * 3 + 2];
    
    // Check if this face has enough visible vertices
    bool isFaceVisible = countVisibleVtsPerFaceCuda(f0, f1, f2, isVis, nTHR, isVvis);
    
    if (isFaceVisible)
    {
        isFvis[f] = true;
    }
}