# geo3dcull — Compiling the Shared Libraries

This guide covers building the CPU and GPU (CUDA) shared libraries on
Windows, Linux, and macOS, plus how to verify the result. It is self-contained:
nothing here depends on a specific machine or environment.

## What is being built

| Library | Source root | Output | Purpose |
| --- | --- | --- | --- |
| **CPU** (OpenMP) | `src/` | `libs/Geo3DCullLib.dll` · `libs/libGeo3DCullLib.so` | Visibility culling on the CPU, multithreaded with OpenMP |
| **GPU** (CUDA) | `cuda/` | `libs/Geo3DCullLibCuda.dll` · `libs/libGeo3DCullLibCuda.so` | Same culling, accelerated on an NVIDIA GPU |

Both expose an `extern "C"` API with the same two core entry points:

- CPU: `testOpenMP()`, `OMP_getVtsCullCamLoS(...)`, `OMP_getVisFcsVtsFromCull(...)`
- GPU: `testCuda()`, `cuda_getVtsCullCamLoS(...)`, `cuda_getVisFcsVtsFromCull(...)`

The Python wrapper dispatches to the right set at runtime based on `use_cuda`,
so only the matching library must be present.

> You do **not** need both. Build only the implementation(s) you will run.
> The GPU build requires an NVIDIA GPU + CUDA toolkit; the CPU build does not.

## Prerequisites

### Common
- A C++17 compiler:
  - **Windows:** Microsoft Visual Studio 2019 or 2022 (MSVC `cl.exe`, x64)
  - **Linux/macOS:** GCC 7+ or Clang 5+
- CMake 3.15+ (for the cross-platform build) — optional if using the Makefile

### CPU-only
- OpenMP:
  - MSVC: built-in (`/openmp`) — no extra install
  - GCC/Clang: included; on some distros install `libomp` if using the
    LLVM runtime (`sudo apt-get install libomp-dev` / `brew install libomp`)

### GPU (CUDA)
- An NVIDIA GPU (compute capability 5.0 or higher recommended)
- NVIDIA driver
- **CUDA Toolkit 12.0+** with `nvcc` on the `PATH`
  - Windows: CUDA toolkit installer (optionally with the VS integration)
  - Linux: `sudo apt-get install nvidia-cuda-toolkit` or the official installer

---

## Option A — CMake (recommended, all platforms)

Works on Windows (MSVC), Linux, and macOS. GPU build is opt-in.

```bash
# CPU only
cmake -S . -B build
cmake --build build --config Release        # add --config only for multi-config generators

# CPU + GPU
cmake -DGPU=ON -S . -B build
cmake --build build --config Release
```

Output: `libs/Geo3DCullLib.dll` (or `libs/libGeo3DCullLib.so` on POSIX) and,
if `-DGPU=ON`, `libs/Geo3DCullLibCuda.dll` (or `.so`).

Target a specific GPU architecture:
```bash
cmake -DGPU=ON -DCMAKE_CUDA_ARCHITECTURES=80 -S . -B build
```

## Option B — Makefile (Linux/macOS)

```bash
make cpu       # builds libs/libGeo3DCullLib[.so|.dylib]
make cuda      # builds libs/libGeo3DCullLibCuda.so (needs nvcc)
make all       # cpu, then cuda if nvcc is available
make test      # build the CPU lib + the C++ test and run it
make python    # pip install ./python into the current environment
make clean     # remove build artefacts
```

Override the compiler or GPU architecture:
```bash
make CXX=clang++
make CUDA_ARCH="-gencode=arch=compute_80,code=sm_80"
```

## Option C — MSVC command line (Windows, no CMake/VS project)

From a **Developer Command Prompt for VS** (or after `vcvarsall.bat x64`):

```bat
set "G3C_ROOT=%CD%"
set "G3C_LIB=%G3C_ROOT%\libs"
mkdir "%G3C_LIB%" 2>nul

:: CPU (Debug CRT)
cl /MDd /EHsc /std:c++17 /openmp ^
   /I"%G3C_ROOT%\src\include" /I"%G3C_ROOT%\src\cpp" ^
   /D GEO3DCULLLIB_EXPORTS ^
   "%G3C_ROOT%\src\cpp\Geo3DCullLib.cpp" "%G3C_ROOT%\src\cpp\Geo3DCullLibUtils.cpp" ^
   /link /DLL /OUT:"%G3C_LIB%\Geo3DCullLib.dll"

:: For Release use /MD /O2 instead of /MDd
```

GPU (CUDA) via `nvcc`:
```bat
nvcc -shared -Xcompiler /MDd, /D GEO3DCULLLIB_CUDA_EXPORTS ^
     -std=c++17 -arch=compute_70,code=sm_70 -arch=compute_80,code=sm_80 ^
     -I"%G3C_ROOT%\cuda\include" -I"%G3C_ROOT%\cuda\src" ^
     "%G3C_ROOT%\cuda\src\Geo3DCullLibCuda.cu" "%G3C_ROOT%\cuda\src\Geo3DCullLibCudaUtils.cu" ^
     -o "%G3C_LIB%\Geo3DCullLibCuda.dll"
```

---

## Verifying a build

**C++ (CPU)** — from a Developer Command Prompt (Windows):
```bat
cl /MDd /EHsc /std:c++17 /openmp /I"src\include" ^
   /Fe:"test\test_program.exe" "test\test_program.cpp" ^
   /link /LIBPATH:"libs" Geo3DCullLib.lib
copy /Y libs\Geo3DCullLib.dll test\ >nul
test\test_program.exe
```

Linux/macOS: `make test` (builds the lib + test and runs it).

**Python** — the wrapper finds the library in `libs/` automatically:
```bash
pip install ./python
python python/tests/test_wrapper.py
```

**GPU** (needs a CUDA device + the GPU library built):
```python
import geo3dcull
w = geo3dcull.Geo3DCullDLL(use_cuda=True)
w.testCuda()
```

---

## Output locations — quick reference

| Platform | CPU | GPU |
| --- | --- | --- |
| Windows | `libs/Geo3DCullLib.dll` | `libs/Geo3DCullLibCuda.dll` |
| Linux/macOS | `libs/libGeo3DCullLib.so` (or `.dylib`) | `libs/libGeo3DCullLibCuda.so` |

The Python wrapper searches `libs/` first (see `README.md` → *Locating the
shared library*), so no further configuration is needed after a build.

---

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `Geo3DCullLib shared library not found` at import | Library not built yet, or not in `libs/`. Build it, or set `GEO3DCULL_LIB_PATH`, or pass `G3Cdll_path=`. |
| `unresolved external symbol` / missing exports at link | Windows build missing `/D GEO3DCULLLIB_EXPORTS` (CPU) or `/D GEO3DCULLLIB_CUDA_EXPORTS` (GPU). |
| `The specified procedure could not be found` from ctypes | Calling the wrong implementation (e.g. CUDA fn from the CPU library). Match `use_cuda` to the library loaded. |
| CRT mismatch (`MSVCP140` / `VCRUNTIME`) | Build the consumer and the DLL with the same CRT (`/MD` or `/MDd`). |
| `nvcc: command not found` | CUDA toolkit not on PATH; open a CUDA/Developer prompt or add `<cuda>/bin`. |
| `unsupported gpu architecture` from nvcc | Set `-arch` / `CMAKE_CUDA_ARCHITECTURES` / `CUDA_ARCH` to your GPU's compute capability. |
| "no kernel image" at runtime | Built for a compute capability lower than the GPU's; rebuild for the correct `-arch`. |
| CPU build not parallel | OpenMP not enabled (`/openmp` on MSVC, `-fopenmp` on GCC/Clang). |
| `#include <omp.h>` fails (MSVC) | `Geo3DCullLib.cpp` uses OpenMP; build with `/openmp` (CMake does this automatically via `find_package(OpenMP)`). |

---

## API symbols (for reference)

### CPU — `Geo3DCullLib`
```c
void testOpenMP(void);
void OMP_getVtsCullCamLoS(double* vts, double* vns, int* V,
                          int* fcs, double* fns, double* vfs, int* F,
                          double* camPos, double* camDir, double* THR, bool* isVis);
void OMP_getVisFcsVtsFromCull(int* fcs, int* F, bool* isVis, int* V,
                              int* nTHR, bool* isFvis, bool* isVvis);
```

### GPU — `Geo3DCullLibCuda`
```c
void testCuda(void);
void cuda_getVtsCullCamLoS(double* vts, double* vns, int* V,
                           int* fcs, double* fns, double* vfs, int* F,
                           double* camPos, double* camDir, double* THR, bool* isVis);
void cuda_getVisFcsVtsFromCull(int* fcs, int* F, bool* isVis, int* V,
                               int* nTHR, bool* isFvis, bool* isVvis);
```

Full parameter documentation lives in `src/include/Geo3DCullLib.h` and
`cuda/include/Geo3DCullLibCuda.h`.
