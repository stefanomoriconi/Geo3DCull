# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:24:32 2025

@author: OGSMORIC

Python wrapper for the ``geo3dcull`` C++/CUDA toolkit. Self-contained and
compatible with both the CPU (OpenMP) and the GPU (CUDA) implementations,
chosen at runtime via ``use_cuda``.
"""

import numpy as np
import ctypes
import time
import os
import scipy

# matplotlib is required ONLY by the visualization helpers
# (``Geometry._showGeo3D`` / ``Geo3DCullDLL._showTest``). It is imported
# defensively so that the core culling calls still work in headless
# environments without matplotlib. Install it with:  pip install matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# point_cloud_utils is required ONLY by Geometry.decimate() (used in the
# runTestLowPoly demo). It is imported defensively so that the rest of this
# wrapper (the culling calls and tests) still works in environments where
# the package is not installed (e.g. the base Python instead of a conda
# env). Install it with:  pip install point-cloud-utils
try:
    import point_cloud_utils as pcu
except ImportError:
    pcu = None

# Ancillary functions, kept here so this wrapper is self-contained.
#
# NOTE on naming: every top-level symbol in this module is prefixed with
# "g3c_" (Geo3DCull) so that -- even if a consumer does
# `from geo3dcull import *` -- nothing here can shadow a common library /
# stdlib name.
def g3c_uvect(v3):
    # retrieve unit-vector: v3 expected array [3,]
    uv3 = np.array(v3)
    if len(v3.shape) == 1:
        uv3 = uv3/np.linalg.norm(uv3)
    else:
        dd = np.linalg.norm(uv3, axis=1)
        uv3 = np.divide(uv3, dd.reshape(dd.shape[0], 1))
    return uv3

def g3c_projct(a3, b3):
    # projection of vector a3 to vector b3: a3 and b3 expected arrays [3,]
    c = np.dot(a3, b3)
    if len(np.shape(a3)) > 1:
        c = np.reshape(c, [np.shape(c)[0], 1])
    return c

class Geometry:
    """
    A self-contained triangle-mesh geometry container. It stores the raw
    vertices and faces and, on demand, derived features (face/vertex normals,
    face centres, average triangle size, centre of mass) plus a mesh decimator.
    Only NumPy is required for the core; SciPy is used for the low-res to
    high-res face mapping and matplotlib (optional) for visualization.
    """

    def __init__(self, vts=None, fcs=None):
        self.vts = vts
        self.fcs = fcs
        self.vn = None
        self.fn = None
        self.fcs_ctr = None
        self.CoM = np.zeros(3, dtype=np.float64)
        self.trisize = 0.0
        self._getFeatures()

    def _getFeatures(self):
        if self.fcs is None or self.vts is None:
            return

        vts = np.asarray(self.vts, dtype=np.float64)
        fcs = np.asarray(self.fcs)
        self.vts = vts
        self.fcs = fcs

        # Centre of mass (mean of vertex coordinates)
        self.CoM = np.mean(vts, axis=0)

        # Compute face centres
        fcs_vts = np.stack((vts[fcs[:, 0], :],
                            vts[fcs[:, 1], :],
                            vts[fcs[:, 2], :]), axis=2)
        self.fcs_ctr = np.mean(fcs_vts, axis=2)

        # Compute face normals
        v1s = vts[fcs[:, 1], :] - vts[fcs[:, 0], :]
        v2s = vts[fcs[:, 2], :] - vts[fcs[:, 1], :]
        self.fn = g3c_uvect(np.cross(v1s, v2s))

        # Compute vertex normals (average of adjacent face normals)
        idx = np.argsort(fcs.flatten()).astype(np.float64)
        idx = np.floor(idx / fcs.shape[1]).astype(np.int64)
        _, cts = np.unique(fcs, return_counts=True)
        fns_vts = self.fn[idx, :]
        fns_vts = np.split(fns_vts, np.cumsum(cts)[:-1])
        self.vn = np.array([g3c_uvect(np.mean(i, axis=0)) for i in fns_vts])

        # Average triangle (edge) size
        e0 = np.linalg.norm(vts[fcs[:, 0], :] - vts[fcs[:, 1], :], axis=1)
        e1 = np.linalg.norm(vts[fcs[:, 1], :] - vts[fcs[:, 2], :], axis=1)
        e2 = np.linalg.norm(vts[fcs[:, 2], :] - vts[fcs[:, 0], :], axis=1)
        self.trisize = float(np.mean(np.concatenate((e0, e1, e2))))

    def stats(self, fullFlag=True):
        print(' ')
        print(' * Geometry Stats*')
        print(' - Vertices: {:d} x {:d}'.format(self.vts.shape[0], self.vts.shape[1]))
        print(' - Faces: {:d} x {:d}'.format(self.fcs.shape[0], self.fcs.shape[1]))
        print(' - Face Size: {:.3f}'.format(self.trisize))
        if fullFlag:
            print(' - CoM coords XYZ: [{:.3f},{:.3f},{:.3f}]'.format(
                self.CoM[0], self.CoM[1], self.CoM[2]))
        return None

    def decimate(self, fcsRate=0.5):
        """
        Decimate (simplify) the triangular mesh in-place, reducing the number
        of faces to roughly ``fcsRate`` of the original count.

        Mirrors ``Geometry.decimate`` from ``geometry_class.py``: it delegates
        Delegates to ``point_cloud_utils.decimate_triangle_mesh`` so the same
        robust,
        Parameters
        ----------
        fcsRate : float, optional
            Fraction of the original face count to keep (0 < fcsRate < 1).
            The default is 0.5.

        Raises
        ------
        ValueError
            If ``fcsRate`` is not strictly within (0, 1).
        """
        fcsRate = float(fcsRate)
        if not (0.0 < fcsRate < 1.0):
            raise ValueError("decimate(): fcsRate must be strictly in (0, 1).")

        if pcu is None:
            raise ImportError(
                "Geometry.decimate() requires the 'point_cloud_utils' package, "
                "which is not available in this Python environment. "
                "Install it with 'pip install point-cloud-utils'.")

        maxFcs = int(fcsRate * self.fcs.shape[0])
        vts_red, fcs_red, _, _ = pcu.decimate_triangle_mesh(self.vts,
                                                            self.fcs,
                                                            max_faces=maxFcs)
        # Assignment
        self.vts = vts_red
        self.fcs = fcs_red
        # Re-initialise derived features (normals, CoM, trisize, ...)
        self._getFeatures()

    def _showGeo3D(self, fig=None, fcsAlpha=0.5, edgeAlpha=0.25, linewidth=0.5,
                   supTitle=None, shadeFlag=False, antiAliasFlag=True):
        if plt is None:
            raise ImportError(
                "matplotlib is required for visualization (Geometry._showGeo3D), "
                "but it is not available in this Python environment. "
                "Install it with 'pip install matplotlib'.")
        # Enable interactive mode
        plt.ion()

        hs = []  # handles to Geometries plotted

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.001)
            ax = fig.axes[0]

        # Displaying Geometry as Triangular Mesh (Patch)
        h0 = ax.plot_trisurf(self.vts[:, 0],
                             self.vts[:, 1],
                             self.vts[:, 2],
                             triangles=self.fcs,
                             color=[0.5, 0.5, 0.5],
                             edgecolor=[0.0, 0.0, 0.0, edgeAlpha],
                             linewidth=linewidth,
                             alpha=fcsAlpha,
                             shade=shadeFlag,
                             antialiased=antiAliasFlag)
        hs.append(h0)

        ax.set_aspect('equal')
        ax.axes.set_xlabel('X-axis')
        ax.axes.set_ylabel('Y-axis')
        ax.axes.set_zlabel('Z-axis')

        if supTitle is not None:
            fig.suptitle(supTitle)

        # Re-drawing the figure
        fig.canvas.draw()

        return fig, hs


class Geo3DCullDLL:
    def __init__(self, G3Cdll_path=None, use_cuda=False):
        # Initialising fields
        self.use_cuda = use_cuda
        self.G3Cdll = self._loadG3Cdll(G3Cdll_path)
        
    # -- Shared-library resolution -----------------------------------------
    # The wrapper loads a compiled shared object. The name and search order are
    # chosen so that a freshly-cloned repo "just works" once the library has
    # been built, without hard-coding any machine-specific path.
    def _default_dll_name(self):
        """Return the shared-library filename for the active implementation."""
        base = "Geo3DCullLibCuda" if self.use_cuda else "Geo3DCullLib"
        return base + ".dll" if os.name == "nt" else "lib" + base + ".so"

    def _candidate_dll_dirs(self):
        """Ordered list of directories to search for the shared library."""
        here = os.path.dirname(os.path.abspath(__file__))
        # .../geo3dcull/python/geo3dcull -> repo root is two levels up
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))

        dirs = []
        # 1) Explicit override: GEO3DCULL_LIB_PATH (a directory, or a file path)
        envp = os.environ.get("GEO3DCULL_LIB_PATH")
        if envp:
            dirs.append(envp if os.path.isdir(envp) else os.path.dirname(envp))
        # 2) Package-local build output
        dirs.append(os.path.join(repo_root, "libs"))
        dirs.append(os.path.join(repo_root, "x64", "Debug"))
        dirs.append(os.path.join(repo_root, "x64", "Release"))
        # 3) Legacy relative locations (current working directory)
        for sub in (("x64", "Debug"), ("x64", "Release"), ("libs",)):
            dirs.append(os.path.join(os.getcwd(), *sub))
        return dirs

    def _resolve_dll(self):
        """Locate the shared library, or return its bare filename as a fallback."""
        name = self._default_dll_name()
        seen = set()
        for d in self._candidate_dll_dirs():
            d = os.path.abspath(d)
            if d in seen:
                continue
            seen.add(d)
            if os.path.isdir(d):
                cand = os.path.join(d, name)
                if os.path.exists(cand):
                    return cand
        return name

    def _loadG3Cdll(self, G3Cdll_path=None):
        if G3Cdll_path is None:
            G3Cdll_path = self._resolve_dll()
            if not os.path.exists(G3Cdll_path):
                raise FileNotFoundError(
                    "Geo3DCullLib shared library not found (looked for "
                    f"'{self._default_dll_name()}').\n"
                    "Build it first (see COMPILING.md), then either:\n"
                    "  - place it in '<repo>/libs/', or\n"
                    "  - set GEO3DCULL_LIB_PATH to the directory that contains it, or\n"
                    "  - pass an explicit path: Geo3DCullDLL(G3Cdll_path=...).")

        G3Cdll = ctypes.CDLL(G3Cdll_path)
        return G3Cdll
    
    def testOpenMP(self):
        """Test OpenMP functionality - only available when using the original DLL"""
        if self.use_cuda:
            raise AttributeError("testOpenMP() is not available when using CUDA version. "
                               "Use testCuda() or initialize with use_cuda=False.")
        try:
            self.G3Cdll.testOpenMP()
        except AttributeError:
            raise AttributeError("testOpenMP() function not found in loaded DLL. "
                               "Make sure you are using the original OpenMP version.")
        
    def testCuda(self):
        """Test CUDA functionality if using CUDA version"""
        if not self.use_cuda:
            raise AttributeError("testCuda() is only available when using CUDA version. "
                               "Initialize with use_cuda=True to enable CUDA functions.")
        try:
            self.G3Cdll.testCuda()
        except AttributeError:
            raise AttributeError("testCuda() function not found in loaded DLL. "
                               "Make sure you are using the CUDA version.")
    
    def getVisibleFcsVtsCull(self, vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR):
        '''
        Main Wrapping Function for the (COMPILED + OpenMP) CULLing process.
        Use this function to compute the set of vertices and faces that are 
        visible in front of a camera.

        Parameters
        ----------
        vts : np.array(), shape: Vx3, with V: num_vertices
            Array of Vertices' COORDINATES of a mesh GEOMETRY in 3D.
        vns : np.array(), shape: Vx3, with V: num_vertices
            Array of Vertices' NORMALS of a mesh GEOMETRY in 3D.
        fcs : np.array(), shape: Fx3, with F: num_faces
            Triangular tessellation of the mesh FACES, int indexes of vertices.
        fns : np.array(), shape: Fx3, with F: num_faces
            Array of Faces' NORMALS of a mesh GEOMETRY in 3D.
        vfs : np.array(), shape: Fx3, with F: num_faces
            Array of Faces' Centers COORDINATES of a mesh GEOMETRY in 3D.
        camPos : np.array(), shape: 3
            Array defining the Camera Position COORDINATES in 3D.
        camDir : np.array(), shape: 3
            Array defining the Camera Orientation DIRECTION in 3D.
        dTHR : np.array(), shape: 1
            Scalar distance THRESHOLD for Neighbouring Faces 
            (usually 1.5 times the average size of a triangle in the mesh).
        bTHR : np.array(), shape: 1
            Scalar integer THRESHOLD for number of VISIBLE VERTICES per FACE.
            This parameter RELAXES the visible BOUNDARY:
                with 1 (MOST RELAXED) and 3 (MOST RESTRICTIVE)
            The default is 3. It can be either 1, 2, or 3.

        Returns
        -------
        isFvis : BOOLEAN array - vector of len(num_faces)
            FLAG indicating face visibility, determined by the CULLing.
        isVvis : BOOLEAN array - vector of len(num_vertices)
            FLAG indicating vertex visibility, determined by the CULLing.

        '''
        # Homogeneising data types -- # Make Sure oder is C-like! (ROW-major)
        vts = vts.astype(np.double, order='C')
        vns = vns.astype(np.double, order='C')
        V = np.array([vts.shape[0]], dtype=np.intc)
        fcs = fcs.astype(np.intc, order='C')
        fns = fns.astype(np.double, order='C')
        vfs = vfs.astype(np.double, order='C')
        F = np.array([fcs.shape[0]], dtype=np.intc)
        camPos = camPos.astype(np.double)
        camDir = camDir.astype(np.double)
        dTHR = dTHR.astype(np.double)
        bTHR = bTHR.astype(np.intc)
        
        # Initialising Output
        isVisTMP = np.ones(vts.shape[0], dtype=np.bool_)

        isFvis = np.ones(fcs.shape[0], dtype=np.bool_)
        isVvis = np.ones(vts.shape[0], dtype=np.bool_)
        
        # Creating Pointers
        vts_ptr = vts.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vns_ptr = vns.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        fcs_ptr = fcs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        fns_ptr = fns.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vfs_ptr = vfs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        F_ptr = F.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        camPos_ptr = camPos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        camDir_ptr = camDir.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dTHR_ptr = dTHR.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        bTHR_ptr = bTHR.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        isVisTMP_ptr = isVisTMP.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        isFvis_ptr = isFvis.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        isVvis_ptr = isVvis.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        
        # Call Initial Function from DLL
        if self.use_cuda:
            # CUDA version function names
            self.G3Cdll.cuda_getVtsCullCamLoS(vts_ptr,
                                             vns_ptr,
                                             V_ptr,
                                             fcs_ptr,
                                             fns_ptr, 
                                             vfs_ptr,
                                             F_ptr,
                                             camPos_ptr,
                                             camDir_ptr,
                                             dTHR_ptr,
                                             isVisTMP_ptr)
            # Call Final Function from DLL
            self.G3Cdll.cuda_getVisFcsVtsFromCull(fcs_ptr,
                                                 F_ptr,
                                                 isVisTMP_ptr,
                                                 V_ptr,
                                                 bTHR_ptr, 
                                                 isFvis_ptr,
                                                 isVvis_ptr)
        else:
            # Original version function names
            self.G3Cdll.OMP_getVtsCullCamLoS(vts_ptr,
                                             vns_ptr,
                                             V_ptr,
                                             fcs_ptr,
                                             fns_ptr, 
                                             vfs_ptr,
                                             F_ptr,
                                             camPos_ptr,
                                             camDir_ptr,
                                             dTHR_ptr,
                                             isVisTMP_ptr)
            # Call Final Function from DLL
            self.G3Cdll.OMP_getVisFcsVtsFromCull(fcs_ptr,
                                                 F_ptr,
                                                 isVisTMP_ptr,
                                                 V_ptr,
                                                 bTHR_ptr, 
                                                 isFvis_ptr,
                                                 isVvis_ptr)
        
        return isFvis, isVvis
    
    def getVisibleFcsVtsCull_CUDA(self, vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR):
        """
        Convenience method to explicitly use CUDA version (alternative to constructor parameter)
        
        This method forces the use of the CUDA implementation even if it wasn't specified
        during initialization.
        """
        # Store original setting
        original_use_cuda = self.use_cuda
        
        # Temporarily enable CUDA
        self.use_cuda = True
        
        try:
            result = self.getVisibleFcsVtsCull(vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR)
            return result
        finally:
            # Restore original setting
            self.use_cuda = original_use_cuda

    def runTest(self, G=[], camPos=[], camDir=[], nsub=2, bTHR=3, stressTestFlag=0, showFlag=1):
        '''
        Run a comprehensive test using an icosphere geometry.

        Parameters
        ----------
        G : GEOMETRY OBJECT from CLASS, optional
            OBJECT defining the Testing Geometry. The default is [].
            An Icosphere will be generated if empty.
        nsub : positive INTEGER, optional
            Number of subdivisions of the input mesh Geometry.
            The default is 2. Higher values will increase face, vertex density.
        bTHR : positive INTEGER, optional
            THRESHOLD for number of VISIBLE VERTICES per FACE.
            This parameter RELAXES the visible BOUNDARY:
                with 1 (MOST RELAXED) and 3 (MOST RESTRICTIVE)
            The default is 3. It can be either 1, 2, or 3.
        stressTestFlag : BOOLEAN or [0, 1], optional
            FLAG to activate the Performance Stress-test: for-loop (30 iters).
            The default is 0.
        showFlag : BOOLEAN or [0, 1], optional
            FLAG to activate visualisation of the face/vertex CULLing results,
            for the given Geometry and for a randomly generated camera.
            The default is 0.

        '''
        # Accept an arbitrary geometry object, or generate an icosphere
        if not isinstance(G, Geometry):
            G = self._genIcoSphere(nsub)

        # Determining a random position and orientation of a point-wise camera
        if len(camPos) == 0:
            camPos = G.CoM.flatten() + np.multiply(np.random.randn(3),
                                                   30.0 * G.trisize)
        if len(camDir) == 0:
            camDir = G.CoM.flatten() - camPos
            camDir = camDir / np.linalg.norm(camDir)

        # Create test data
        vts = G.vts
        vns = G.vn
        fcs = G.fcs
        fns = G.fn
        vfs = G.fcs_ctr

        dTHR = np.array([G.trisize * 1.5], dtype=np.float64)  # Distance threshold
        bTHR = np.array([bTHR], dtype=np.int32)      # Boundary threshold

        print(f"Running test with {vts.shape[0]} vertices and {fcs.shape[0]} faces")

        if stressTestFlag:
            print("Starting performance stress test (30 iterations)...")
            G.stats(fullFlag=0)
            start_time = time.time()
            for i in range(30):
                isFvis, isVvis = self.getVisibleFcsVtsCull(vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR)
            end_time = time.time()
            print(f"Average time per iteration: {(end_time - start_time) * 1000 / 30:.2f} ms")
        else:
            # Run single test
            isFvis, isVvis = self.getVisibleFcsVtsCull(vts, vns, fcs, fns, vfs, camPos, camDir, dTHR, bTHR)
            
            print(f"Test completed. Visible faces: {np.sum(isFvis)}, Visible vertices: {np.sum(isVvis)}")
            
            if showFlag:
                # Display basic results
                print("Face visibility (first 10):", isFvis[:10])
                print("Vertex visibility (first 10):", isVvis[:10])
                self._showTest(G, camPos, camDir, isFvis, isVvis, 'FaceCULLING Icosphere')

    def runTestLowPoly(self, G_highres=[], camPos=[], camDir=[], nsub=3,
                       bTHR=3, decimRate=0.1, showFlag=1):
        '''
        This function runs a demo for CULLING, where the High-resolution
        mesh is decimated first into a Low-Resolution version, the CULLING is
        performed on the Low-Resolution one (fast), and the resulting face- and
        vertex-CULLING is mapped back onto the High-resolution instance.
        Plots are shown to highlight visual differences, also stats (precision,
        recall, accuracy) for both CULLED faces and vertices are provided with
        computational performance.

        On complex meshes, it may be worthwile tuning the bTHR parameter to
        increase outcomes!

        Parameters
        ----------
        G_highres : GEOMETRY OBJECT, optional
            OBJECT defining the HIGH-RESOLUTION Testing Geometry.
            The default is []. An Icosphere (nsub subdivisions) will be
            generated if empty.
        camPos : np.array(3), optional
            Camera position in 3D. Randomly generated if empty.
        camDir : np.array(3), optional
            Camera orientation direction in 3D. Randomly generated if empty.
        nsub : positive INTEGER, optional
            Number of subdivisions of the input mesh Geometry (used only when
            ``G_highres`` is empty). The default is 3.
        bTHR : positive INTEGER, optional
            THRESHOLD for number of VISIBLE VERTICES per FACE.
            This parameter RELAXES the visible BOUNDARY:
                with 1 (MOST RELAXED) and 3 (MOST RESTRICTIVE)
            The default is 3. It can be either 1, 2, or 3.
        decimRate : float in (0, 1), optional
            Fraction of the high-resolution faces to keep in the
            low-resolution mesh produced by ``Geometry.decimate``.
            The default is 0.1 (keep 10 % of the faces).
        showFlag : BOOLEAN or [0, 1], optional
            FLAG to activate visualisation of the face/vertex CULLing results,
            for the given Geometry and for a randomly generated camera.
            The default is 1.

        Returns
        -------
        G_highres : GEOMETRY OBJECT
            High-resolution geometry used for the test
        camPos : np.array(3) - vector of double
            Position of the Camera in 3D (Randomly generated)
        camDir : np.array(3) - vector of double
            Orientation of the Camera in 3D (Randomly generated)
        isFvis : BOOLEAN array - vector of len(num_faces)
            FLAG indicating face visibility, determined by the CULLing.
        isVvis : BOOLEAN array - vector of len(num_vertices)
            FLAG indicating vertex visibility, determined by the CULLing.

        '''
        # Accept an arbitrary high-resolution geometry, or generate an icosphere
        if not isinstance(G_highres, Geometry):
            G_highres = self._genIcoSphere(nsub=nsub)

        # Determining a random position and orientation of a point-wise camera
        if len(camPos) == 0:
            camPos = G_highres.CoM.flatten() + np.multiply(np.random.randn(3),
                                                           30.0 * G_highres.trisize)
        if len(camDir) == 0:
            camDir = G_highres.CoM.flatten() - camPos
            camDir = camDir / np.linalg.norm(camDir)

        # Decimate a COPY of the high-resolution mesh into the low-resolution one
        G_lowres = Geometry(vts=G_highres.vts, fcs=G_highres.fcs)
        G_lowres.decimate(decimRate)

        # Mapping indices of GEOMETIES: from High-Density FACES TO Low-Density FACES
        HRfcs2LRfcs = self._mapGeo3D_HighResFcs2LowResFcs(G_highres.fcs_ctr,
                                                          G_lowres.fcs_ctr)
        
        # Initialising Inputs to the Function for CULLing
        LR_vts = G_lowres.vts # Vertices
        LR_vns = G_lowres.vn # Vertices' normals
        LR_fcs = G_lowres.fcs # Faces of triangulation
        LR_fns = G_lowres.fn # Faces normals
        LR_vfs = G_lowres.fcs_ctr # Faces centres coordinates
        LR_dTHR = np.array(1.5 * G_lowres.trisize) # Threshold of 1.5 the avg Triangle size
        bTHR = np.array([bTHR])
        
        HR_vts = G_highres.vts # Vertices
        HR_vns = G_highres.vn # Vertices' normals
        HR_fcs = G_highres.fcs # Faces of triangulation
        HR_fns = G_highres.fn # Faces normals
        HR_vfs = G_highres.fcs_ctr # Faces centres coordinates
        HR_dTHR = np.array(1.5 * G_highres.trisize) # Threshold of 1.5 the avg Triangle size
        
        # Low-Resolution CULLING
        print('* Low-Resolution *')
        G_lowres.stats(fullFlag=0)
        LR_t0 = time.time()
        LR_isFvis, LR_isVvis = self.getVisibleFcsVtsCull(LR_vts, LR_vns, 
                                                         LR_fcs, LR_fns, LR_vfs,
                                                         camPos, camDir,
                                                         LR_dTHR, bTHR)
        
        HR_isFvis_LRmap, HR_isVvis_LRmap = self._mapGeo3D_VisibFcsVts(LR_isFvis,
                                                                      HRfcs2LRfcs,
                                                                      G_highres)
        LR_time = time.time() - LR_t0
        
        # High-Resolution CULLING
        print('* High-Resolution (GT)*')
        G_highres.stats(fullFlag=0)
        HR_t0 = time.time()
        HR_isFvis, HR_isVvis = self.getVisibleFcsVtsCull(HR_vts, HR_vns, 
                                                         HR_fcs, HR_fns, HR_vfs,
                                                         camPos, camDir,
                                                         HR_dTHR, bTHR)
        HR_time = time.time() - HR_t0
        print('FaceCULLING Low-Resolution - Elapsed Time: {:.3f}s'.format(LR_time))
        print('FaceCULLING High-Resolution Ground-Truth - Elapsed Time: {:.3f}s'.format(HR_time))
        
        if showFlag:
            self._showTest(G_lowres, camPos, camDir, LR_isFvis, LR_isVvis, 'FaceCULLING Low-Resolution')
            self._showTest(G_highres, camPos, camDir, HR_isFvis_LRmap, HR_isVvis_LRmap, 'FaceCULLING High-Resolution MAP from Low-Resolution')
            self._showTest(G_highres, camPos, camDir, HR_isFvis, HR_isVvis, 'FaceCULLING High-Resolution Ground-Truth')
        
        # Print Stats Face and Vertices: precision, recall, accuracy
        fTP = np.sum(np.logical_and(HR_isFvis_LRmap, HR_isFvis))
        fFP = np.sum(np.logical_and(HR_isFvis_LRmap, np.logical_not(HR_isFvis)))
        fFN = np.sum(np.logical_and(np.logical_not(HR_isFvis_LRmap), HR_isFvis))
        fTN = np.sum(np.logical_and(np.logical_not(HR_isFvis_LRmap), np.logical_not(HR_isFvis)))
        
        vTP = np.sum(np.logical_and(HR_isVvis_LRmap, HR_isVvis))
        vFP = np.sum(np.logical_and(HR_isVvis_LRmap, np.logical_not(HR_isVvis)))
        vFN = np.sum(np.logical_and(np.logical_not(HR_isVvis_LRmap), HR_isVvis))
        vTN = np.sum(np.logical_and(np.logical_not(HR_isVvis_LRmap), np.logical_not(HR_isVvis)))
        
        Fpre = np.divide(fTP, fTP + fFP)
        Frec = np.divide(fTP, fTP + fFN)
        Facc = np.divide(fTP + fTN, fTP + fFP + fFN + fTN)
        
        Vpre = np.divide(vTP, vTP + vFP)
        Vrec = np.divide(vTP, vTP + vFN)
        Vacc = np.divide(vTP + vTN, vTP + vFP + vFN + vTN)
        
        print('FaceCULLING Low-Res -> High-Res Stats')
        print('Faces: precision = {:.3f} , recall = {:.3f}, accuracy = {:.3f}'.format(Fpre, Frec, Facc))
        print('Vertices: precision = {:.3f} , recall = {:.3f}, accuracy = {:.3f}'.format(Vpre, Vrec, Vacc))
        
        return G_highres, camPos, camDir, HR_isFvis_LRmap, HR_isVvis_LRmap
    
    def _mapGeo3D_HighResFcs2LowResFcs(self, HR_fcs_ctr, LR_fcs_ctr):
        HRfcs2LRfcs = np.argmin(scipy.spatial.distance.cdist(HR_fcs_ctr,
                                                             LR_fcs_ctr),
                                axis=1)
        return HRfcs2LRfcs
    
    def _mapGeo3D_VisibFcsVts(self, LR_isFvis, HRfcs2LRfcs, Geo3D_HR):
        HR_isFvis = np.isin(HRfcs2LRfcs, np.argwhere(LR_isFvis))
        HR_isVvis = np.isin(np.array(range(Geo3D_HR.vts.shape[0])),
                            np.array(list(set(Geo3D_HR.fcs[HR_isFvis,
                                                           :].flatten()))))
        
        return HR_isFvis, HR_isVvis
    
    def _showTest(self, G, camPos, camDir, isFvis, isVvis, title=[]):
        fig = G._showGeo3D()[0]
        ax = fig.get_axes()[0]
        ax.scatter3D(camPos[0],camPos[1],camPos[2], color='k')
        ax.quiver3D(camPos[0],camPos[1],camPos[2],
                    camDir[0],camDir[1],camDir[2], color='k')
        ax.set_aspect('equal')
        ax.plot_trisurf(G.vts[:, 0], G.vts[:, 1], G.vts[:, 2],
                        triangles = G.fcs[isFvis,:],
                        color=[1.0, 1.0, 0.0],
                        edgecolor=[1.0, 1.0, 0.3, 1.0],
                        linewidth=2.0,
                        alpha=1.0)
        ax.scatter3D(G.vts[isVvis.flatten(),0],
                     G.vts[isVvis.flatten(),1],
                     G.vts[isVvis.flatten(),2],
                     marker='o', color='b')
        
        if len(title) > 0:
            ax.set_title(title)

    def _genIcoSphere(self, nsub=2):
        """
        Generate an icosphere geometry with specified subdivisions.
        
        Parameters
        ----------
        nsub : int, optional
            Number of subdivisions. Default is 2.
            
        Returns
        -------
        Geometry object with the generated icosphere
        """
        # Create icosahedron vertices
        t = (1 + np.sqrt(5.0)) / 2
        vts = np.array([[-1, t, 0], # v1
                        [ 1, t, 0], # v2
                        [-1,-t, 0], # v3
                        [ 1,-t, 0], # v4
                        [ 0,-1, t], # v5
                        [ 0, 1, t], # v6
                        [ 0,-1,-t], # v7
                        [ 0, 1,-t], # v8
                        [ t, 0,-1], # v9
                        [ t, 0, 1], # v10
                        [-t, 0,-1], # v11
                        [-t, 0, 1]]) # v12
        
        # Normalize vertices to unit vectors        
        vts = vts/np.linalg.norm(vts, axis=1, keepdims=1)
        
        # Define faces (triangles)
        fcs = np.array([[ 0, 11,  5], # f1
                        [ 0,  5,  1], # f2
                        [ 0,  1,  7], # f3
                        [ 0,  7, 10], # f4
                        [ 0, 10, 11], # f5
                        [ 1,  5,  9], # f6
                        [ 5, 11,  4], # f7
                        [11, 10,  2], # f8
                        [10,  7,  6], # f9
                        [ 7,  1,  8], # f10
                        [ 3,  9,  4], # f11
                        [ 3,  4,  2], # f12
                        [ 3,  2,  6], # f13
                        [ 3,  6,  8], # f14
                        [ 3,  8,  9], # f15
                        [ 4,  9,  5], # f16
                        [ 2,  4, 11], # f17
                        [ 6,  2, 10], # f18
                        [ 8,  6,  7], # f19
                        [ 9,  8,  1]], dtype=np.uint64)# f20
        
        # Subdivide if needed
        if nsub > 0:
            vts, fcs = self._subdTriFaces(vts, fcs, nsub=nsub)
        
        # Create and return geometry object
        G = Geometry(vts, fcs)
        return G

    def _subdTriFaces(self, vts, fcs, nsub):# OK
        """Subdivide triangular faces recursively"""
        if nsub == 0:
            return vts, fcs
        
        for sbd in range(0, nsub):
            # Initialising subdivided Faces (ONLY Triangles)
            sbd_fcs = np.zeros([fcs.shape[0]*4, 3])
            
            for ff in range(0, fcs.shape[0]): # for each triangular face    
                # Select the i-th Trianglular face
                fc = fcs[ff, :]

                # Calculate the mid points (add new points to v)
                a, vts = self._appendNormMidPoint(fc[0], fc[1], vts)
                b, vts = self._appendNormMidPoint(fc[1], fc[2], vts)
                c, vts = self._appendNormMidPoint(fc[2], fc[0], vts)
                    
                # Generating new subdivision triangles
                nfc = np.array([[fc[0], a, c],
                                [fc[1], b, a],
                                [fc[2], c, b],
                                [    a, b, c]])
                    
                # Replacing Triangle with subdivision
                idx = list(range((4*ff), (4*(ff+1))))
                sbd_fcs[idx, :] = nfc
                
            # Updating Faces with the Subdivided Faces
            fcs = np.array(sbd_fcs, dtype=np.uint64)  
        
        # Removing duplicate vertices
        vts_unq, vts_idx = np.unique(vts.round(decimals=6), axis=0, return_inverse=1)
        # Re-assigning faces to trimmed vertex list and remove duplicate faces
        fcs_unq = np.zeros(fcs.shape, dtype=np.uint64)
        for vidx in range(0, len(vts_idx)):
            fcs_unq[fcs == vidx] = vts_idx[vidx]
        fcs_unq = np.unique(fcs_unq, axis=0)
        
        return vts_unq, fcs_unq
        
    def _appendNormMidPoint(self, idx0, idx1, vts):# OK
        """Append a normalized midpoint between two vertices"""
        # Retrieve vertices 
        v0 = vts[idx0, :]
        v1 = vts[idx1, :]
        
        # New length-normalised Mid-Point
        v2mod = np.mean([np.linalg.norm(v0), np.linalg.norm(v1)])
        v2vec = g3c_uvect(np.mean([v0, v1], axis=0))
        v2 = np.reshape(v2vec, [1, 3])*v2mod
        
        # Concatenate Mid-Point
        vts = np.concatenate((vts, v2), axis=0)
        
        idx = vts.shape[0] - 1
        return idx, vts

# Additional utility functions for CUDA support
class Geo3DCullDLL_CUDA(Geo3DCullDLL):
    """
    Specialized class for CUDA implementation.
    This is an alternative approach to using use_cuda=True in the main class.
    """
    def __init__(self, G3Cdll_path=None):
        super().__init__(G3Cdll_path, use_cuda=True)
        
    def testCuda(self):
        """Override test method for explicit CUDA testing"""
        self.G3Cdll.testCuda()