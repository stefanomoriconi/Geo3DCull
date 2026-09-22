// pch.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// OpenMP is optional: it is enabled automatically by the MSVC project file
// (/openmp) and by -fopenmp in the Makefile. Guard the include so that a
// build without OpenMP support (e.g. some toolchains) still compiles; the
// #pragma omp directives are then simply ignored.
#if defined(_OPENMP)
#include <omp.h>
#endif