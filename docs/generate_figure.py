"""
Generate the visibility-culling figure used in the top-level README.

Uses the library's own `runTest` demo helper (an icosphere + a fixed camera
pose) so the figure is a faithful, reproducible illustration of real, tested
behavior. Run against a built CPU library:

    GEO3DCULL_LIB_PATH=../libs python docs/generate_figure.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "python"))

from geo3dcull import Geo3DCullDLL  # noqa: E402

OUT_DIR = os.path.join(_HERE, "images")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    np.random.seed(7)
    w = Geo3DCullDLL(use_cuda=False)
    G = w._genIcoSphere(nsub=3)

    cam_pos = G.CoM.flatten() + np.array([2.6, 1.4, 1.9]) * G.trisize * 12
    cam_dir = (G.CoM.flatten() - cam_pos)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    w.runTest(G=G, camPos=cam_pos, camDir=cam_dir, bTHR=3, showFlag=1)

    fig = plt.gcf()
    fig.set_size_inches(7, 6)
    out = os.path.join(OUT_DIR, "geo3dcull_visibility.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
