#include "pch.h"
#include "Geo3DCullLibUtils.h"
#include <math.h>

void getLineDir(double p0L[3], double p1L[3], double nnL[3])
{
    double d0[3];
    sbt3(p1L, p0L, d0);
    uvect3D(d0, nnL);
}

double normL2(double p0[3])
{
    return sqrt(pow(p0[0], 2.0) + pow(p0[1], 2.0) + pow(p0[2], 2));
}

double prjct3D(double p0[3], double p1[3])
{
    return p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2];
}

void cross3D(double u[3], double v[3], double w[3])
{
    w[0] = u[1] * v[2] - u[2] * v[1];
    w[1] = u[2] * v[0] - u[0] * v[2];
    w[2] = u[0] * v[1] - u[1] * v[0];
}

void uvect3D(double p0[3], double n0[3])
{
    double L2 = normL2(p0);
    n0[0] = p0[0] / L2;
    n0[1] = p0[1] / L2;
    n0[2] = p0[2] / L2;
}

void add3(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] + v1[0];
    v2[1] = v0[1] + v1[1];
    v2[2] = v0[2] + v1[2];
}

void sbt3(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] - v1[0];
    v2[1] = v0[1] - v1[1];
    v2[2] = v0[2] - v1[2];
}

void mlt3(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] * v1[0];
    v2[1] = v0[1] * v1[1];
    v2[2] = v0[2] * v1[2];
}

void div3(double v0[3], double v1[3], double v2[3])
{
    v2[0] = v0[0] / v1[0];
    v2[1] = v0[1] / v1[1];
    v2[2] = v0[2] / v1[2];
}

double normL2_pt2line(double pt[3], double ptL[3], double nnL[3])
{
    double ptD[3];
    sbt3(pt, ptL, ptD);
    double w[3];
    cross3D(ptD, nnL, w);
    double d1 = normL2(w);
    double d2 = normL2(nnL);
    return d1 / d2;
}

void sectLineWTri(double p0L[3], double p1L[3], double t0[3], double t1[3], double t2[3], double pLP[3])
{
    double segL[3];
    double negL[3];
    double d0[3];
    double d1[3];
    double w[3];
    double d2[3];
    sbt3(p1L, p0L, segL); //segment of the Line in 3D
    negL[0] = -1.0 * segL[0];
    negL[1] = -1.0 * segL[1];
    negL[2] = -1.0 * segL[2];
    sbt3(t1, t0, d0);
    sbt3(t2, t0, d1);
    cross3D(d0, d1, w);
    sbt3(p0L, t0, d2);
    double t_N = prjct3D(w, d2);
    double t_D = prjct3D(negL, w);
    pLP[0] = p0L[0] + (segL[0] * (t_N / t_D));
    pLP[1] = p0L[1] + (segL[1] * (t_N / t_D));
    pLP[2] = p0L[2] + (segL[2] * (t_N / t_D));
}

bool isPtInTri(double p0[3], double t0[3], double t1[3], double t2[3])
{
    double e0[3];
    double e1[3];
    double e2[3];

    sbt3(t2, t0, e0);
    sbt3(t1, t0, e1);
    sbt3(p0, t0, e2);

    double d00 = prjct3D(e0, e0);
    double d01 = prjct3D(e0, e1);
    double d02 = prjct3D(e0, e2);
    double d11 = prjct3D(e1, e1);
    double d12 = prjct3D(e1, e2);

    double iD = 1.0 / ((d00 * d11) - (d01 * d01));
    double u = ((d11 * d02) - (d01 * d12)) * iD;
    double v = ((d00 * d12) - (d01 * d02)) * iD;

    return (u >= 0.0) && (v >= 0.0) && ((u + v) <= 1.0);
}

bool countVisibleVtsPerFace(int f0, int f1, int f2, bool* isVis, int nTHR, bool* isVvis)
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