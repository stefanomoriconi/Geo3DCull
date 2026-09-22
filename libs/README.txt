This directory (libs/) is where the compiled shared libraries are written.

  - CPU:  libGeo3DCullLib[.so|.dylib]
  - GPU:  libGeo3DCullLibCuda.so      (Windows: Geo3DCullLibCuda.dll)

Build them with:

    make cpu     # Linux / macOS
    make cuda    # requires NVIDIA GPU + CUDA toolkit

On Windows, point the build output here (see COMPILING.md), or set the
GEO3DCULL_LIB_PATH environment variable to the directory that contains the
library.

This folder is intentionally empty in the repository (it is git-ignored) so
that every clone starts from source.
