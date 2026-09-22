#pragma once

// Utility functions for Geo3DCullLib
#include <cmath>

/* Function Declarations */

void getLineDir(double* p0L, double* p1L, double* nnL); // Compute the Line unit vector (direction) from two input points in 3D
// INPUTs: 
// p0L: (1x3 double) source point of the line in 3D
// p1L: (1x3 double) target point of the line in 3D
// OUTPUTs:
// nnL: (1x3 double) unit vector indicating the direction of the line in 3D

double normL2(double* v3); // Compute the L2-norm (Euclidean Distance, or Magnitude) of a vector in 3D
// INPUTs:
// v3: (1x3 double) vector in 3D
// OUTPUTs:
// L2: (1x1 double) scalar magnitude of the vector (L2-norm)

double prjct3D(double* p0, double* p1); // Compute the projection (dot product) of two vectors in 3D
// INPUTs:
// p0: (1x3 double) first vector in 3D
// p1: (1x3 double) second vector in 3D
// OUTPUTs:
// p: (1x1 double) projection value of the two vectors in 3D

void cross3D(double* u, double* v, double* w); // Compute the cross product of two vectors in 3D
// INPUTs:
// u: (1x3 double) first vector in 3D
// v: (1x3 double) second vector in 3D
// OUTPUTs:
// w: (1x3 double) resulting vector as cross product between the two input vectors in 3D

void uvect3D(double* p0, double* n0); // Compute the unit vector of the input in 3D
// INPUTs:
// p0: (1x3 double) vector in 3D
// OUTPUTs:
// n0: (1x3 double) unit vector in 3D

// Element-wise Operations on vectors in 3D
void add3(double v0[3], double v1[3], double v2[3]); // Addition
void sbt3(double v0[3], double v1[3], double v2[3]); // Subtraction
void mlt3(double v0[3], double v1[3], double v2[3]); // Multiplication
void div3(double v0[3], double v1[3], double v2[3]); // Division

double normL2_pt2line(double* pt, double* ptL, double* nnL); // Compute the orthogonal L2-distance (Euclidean) of a point towards a line in 3D
// INPUTs:
// pt: (1x3 double) sample point in 3D
// ptL: (1x3 double) any point belonging to the line in 3D
// nnL: (1x3 double) unit vector indicating the direction of the line in 3D
// OUTPUTs:
// d: (1x1 double) distance (orthogonal segment) of the input point towards the line in 3D

void sectLineWTri(double* p0L, double* p1L, double* t0, double* t1, double* t2, double* pLP); // Compute intersection of a line with a triangle
// INPUTs:
// p0L: (1x3 double) source point of the line in 3D
// p1L: (1x3 double) target point of the line in 3D
// t0, t1, t2: (1x3 double) vertices of the triangle in 3D
// OUTPUTs:
// pLP: (1x3 double) intersection point between the line and triangle

bool isPtInTri(double* p0, double* t0, double* t1, double* t2); // Check if a point is inside a triangle
// INPUTs:
// p0: (1x3 double) point in 3D to check
// t0, t1, t2: (1x3 double) vertices of the triangle in 3D
// OUTPUTs:
// true if point is inside triangle, false otherwise

bool countVisibleVtsPerFace(int f0, int f1, int f2, bool* isVis, int nTHR, bool* isVvis); // Count visible vertices per face for boundary relaxation