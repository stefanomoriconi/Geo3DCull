# ============================================================================
#  geo3dcull -- Top-level Makefile
# ----------------------------------------------------------------------------
#  Builds the CPU and (optionally) GPU shared libraries into ./libs/.
#  This Makefile is for Linux/macOS. On Windows, use the CMake generator or
#  build each directory's sources with the MSVC command line (see COMPILING.md).
#
#  Targets:
#      make cpu       # build libGeo3DCullLib[.so|.dylib]   (default)
#      make cuda      # build libGeo3DCullLibCuda.so         (needs nvcc + GPU)
#      make all       # build CPU, then CUDA if nvcc is available
#      make test      # build the CPU lib + the C++ test and run it
#      make python    # install the Python wrapper into the current env (pip)
#      make clean     # remove build artefacts
#
#  Optional overrides:
#      make CXX=clang++
#      make CUDA_ARCH="-gencode=arch=compute_80,code=sm_80"
# ============================================================================

# ---- Toolchain -------------------------------------------------------------
CXX      ?= g++
NVCC     ?= nvcc

# ---- Layout ----------------------------------------------------------------
LIBS      := libs
CPU_SRC   := src/cpp
CPU_INC   := src/include
CUDA_SRC  := cuda/src
CUDA_INC  := cuda/include
TEST_SRC  := test

# Platform-specific shared-library suffix
UNAME_S   := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(UNAME_S),Darwin)
    EXT   := dylib
else
    EXT   := so
endif

CPU_LIB   := $(LIBS)/libGeo3DCullLib.$(EXT)
CUDA_LIB  := $(LIBS)/libGeo3DCullLibCuda.so

# ---- Flags -----------------------------------------------------------------
CXXFLAGS  := -std=c++17 -fPIC -O2 -Wall -Wextra -fopenmp \
             -I$(CPU_INC) -I$(CPU_SRC) -D GEO3DCULLLIB_EXPORTS

CUDA_ARCH ?= -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80
NVCCFLAGS := -std=c++17 -O3 -Xcompiler -fPIC \
             $(CUDA_ARCH) -I$(CUDA_INC) -I$(CUDA_SRC) \
             -DGEO3DCULLLIB_CUDA_EXPORTS

.PHONY: all cpu cuda test python clean

all: cpu
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "nvcc found -- building CUDA library too"; \
		$(MAKE) cuda; \
	else \
		echo "nvcc not found -- skipping CUDA build (run 'make cuda' with the CUDA toolkit installed)"; \
	fi

cpu:
	@mkdir -p $(LIBS)
	$(CXX) $(CXXFLAGS) \
		$(CPU_SRC)/Geo3DCullLib.cpp $(CPU_SRC)/Geo3DCullLibUtils.cpp \
		-shared -o $(CPU_LIB)
	@echo "Built $(CPU_LIB)"

cuda:
	@mkdir -p $(LIBS)
	$(NVCC) $(NVCCFLAGS) \
		$(CUDA_SRC)/Geo3DCullLibCuda.cu $(CUDA_SRC)/Geo3DCullLibCudaUtils.cu \
		-shared -o $(CUDA_LIB)
	@echo "Built $(CUDA_LIB)"

test: cpu
	$(CXX) $(CXXFLAGS) $(TEST_SRC)/test_program.cpp \
		-L$(LIBS) -lGeo3DCullLib -Wl,-rpath,'$$ORIGIN/..,$(CURDIR)/$(LIBS)' \
		-o $(TEST_SRC)/test_program
	@cp -f $(CPU_LIB) $(TEST_SRC)/ 2>/dev/null || true
	@$(TEST_SRC)/test_program

python:
	pip install ./python

clean:
	rm -rf $(LIBS)
	rm -f $(TEST_SRC)/test_program
	rm -f $(CPU_LIB) $(CUDA_LIB)
