#!/usr/bin/env python3
"""
Test script for the refactored Python wrapper of Geo3DCullLib.

This script demonstrates how to use both CPU and CUDA versions of the library.
"""

import numpy as np
from geo3dcull import Geo3DCullDLL

def create_test_geometry():
    """Create a simple test geometry (triangle)"""
    # Create vertices for a simple triangle
    vts = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1  
        [0.0, 1.0, 0.0],  # Vertex 2
    ], dtype=np.float64)
    
    # Create faces (triangle)
    fcs = np.array([
        [0, 1, 2]  # Face with vertices 0, 1, 2
    ], dtype=np.int32)
    
    # Compute normals for vertices and faces
    vns = np.array([
        [0.0, 0.0, 1.0],  # Normal for vertex 0
        [0.0, 0.0, 1.0],  # Normal for vertex 1
        [0.0, 0.0, 1.0],  # Normal for vertex 2
    ], dtype=np.float64)
    
    fns = np.array([
        [0.0, 0.0, 1.0]   # Normal for face 0
    ], dtype=np.float64)
    
    # Compute face centers
    vfs = np.array([
        [0.33333333, 0.33333333, 0.0]  # Center of face 0
    ], dtype=np.float64)
    
    return vts, vns, fcs, fns, vfs

def main():
    print("Testing Geo3DCullLib Python wrapper")
    print("=" * 50)
    
    # Create test geometry
    vts, vns, fcs, fns, vfs = create_test_geometry()
    
    # Camera parameters
    camPos = np.array([0.0, 0.0, 2.0], dtype=np.float64)  # Camera position
    camDir = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Camera direction (looking down)
    dTHR = np.array([0.5], dtype=np.float64)  # Distance threshold
    bTHR = np.array([3], dtype=np.int32)      # Boundary threshold
    
    print("Test geometry created:")
    print(f"Vertices shape: {vts.shape}")
    print(f"Faces shape: {fcs.shape}")
    print(f"Camera position: {camPos}")
    print(f"Camera direction: {camDir}")
    
    try:
        # Test with original CPU implementation
        print("\n1. Testing with CPU version:")
        cpu_wrapper = Geo3DCullDLL(use_cuda=False)
        cpu_wrapper.testOpenMP()
        
        # Run culling test
        isFvis, isVvis = cpu_wrapper.getVisibleFcsVtsCull(
            vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR
        )
        
        print(f"Face visibility results: {isFvis}")
        print(f"Vertex visibility results: {isVvis}")
        
        # Test that CUDA function is not available in CPU version
        print("\n2. Testing function availability - CPU version should not allow CUDA calls:")
        try:
            cpu_wrapper.testCuda()
            print("ERROR: testCuda() should not be available in CPU version")
        except AttributeError as e:
            print(f"Correctly blocked: {e}")
        
    except Exception as e:
        print(f"CPU version test failed: {e}")
    
    try:
        # Test with CUDA implementation (if available)
        print("\n3. Testing with CUDA version:")
        cuda_wrapper = Geo3DCullDLL(use_cuda=True)
        cuda_wrapper.testCuda()
        
        # Run culling test
        isFvis, isVvis = cuda_wrapper.getVisibleFcsVtsCull(
            vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR
        )
        
        print(f"Face visibility results: {isFvis}")
        print(f"Vertex visibility results: {isVvis}")
        
        # Test that OpenMP function is not available in CUDA version
        print("\n4. Testing function availability - CUDA version should not allow OpenMP calls:")
        try:
            cuda_wrapper.testOpenMP()
            print("ERROR: testOpenMP() should not be available in CUDA version")
        except AttributeError as e:
            print(f"Correctly blocked: {e}")
        
    except Exception as e:
        print(f"CUDA version test failed: {e}")
        print("This is expected if CUDA DLL is not built or available.")

if __name__ == "__main__":
    main()