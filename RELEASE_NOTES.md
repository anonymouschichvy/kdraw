# Release Notes: KDRAW v0.1.0 🚀

We are excited to announce the initial release of **KDRAW (v0.1.0)**! 

KDRAW is a high-precision topological centerline vectorizer that converts raster graphics into optimized, smooth single-stroke SVGs. It is designed specifically for CNC plotters, laser cutters, and CAM software.

---

## 🌟 Key Features

* **🧩 Graph-Based Skeleton Tracing**: Traces skeletons as topological graphs (nodes and edges) to prevent junction distortion and line splits.
* **🛡️ Isolated Path Safety (i-Dot Preservation)**: Intelligently distinguishes side spurs (noise) from isolated paths, ensuring punctuation like colons, periods, and the dots of `i` are preserved.
* **🏎️ TSP Pen-Travel Heuristic**: Solves the Travelling Salesperson Problem (TSP) on the toolpath sequence to save up to **98% of pen-up travel distance**.
* **🌀 Chaikin & Laplacian Curve Fitting**: Rounds out path coordinates using subdivision curves for smooth, organic strokes without shrinkage.
* **🔎 4x Upscaled Anti-Aliasing**: Interpolates and smooths boundaries on low-resolution inputs before skeletonization.

---

## 🛠️ Packaging & CI/CD Updates in v0.1.0
To prepare the package for public distribution:
* **CLI Module Refactor**: Standardized the command runner into `kdraw.cli` to package it cleanly while keeping `main.py` as a local wrapper.
* **Modern Package Config**: Added PEP 621-compliant `pyproject.toml` defining package dependencies (`numpy`, `pillow`, `opencv-python`, `scikit-image`) and mapping the CLI script.
* **GitHub Actions integration**: Added `.github/workflows/publish.yml` to support tokenless PyPI Trusted Publishing triggered automatically on GitHub releases.

---

## 📦 Installation & Usage

Install the release directly from PyPI (once published):
```bash
pip install kdraw
```

For smooth Bezier-curve vectorization support:
```bash
pip install kdraw[smooth]
```

Run the vectorizer command:
```bash
kdraw input.jpg output.svg --centerline --no-adaptive
```
