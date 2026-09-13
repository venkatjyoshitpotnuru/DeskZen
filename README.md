# DeskZen — Smart Workspace Ergonomics & Clutter Diagnostic Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Computer Vision](https://img.shields.io/badge/Model-YOLO11--Seg-brightgreen.svg)](https://docs.ultralytics.com/)
[![Deployment](https://img.shields.io/badge/Live-Render-emerald.svg)](https://deskzen-ubmk.onrender.com)

DeskZen is an AI-powered computer vision application designed to evaluate, quantify, and improve workspace cleanliness and ergonomic organization. Using real-time instance segmentation and a 2D Gaussian visual attention field, DeskZen converts raw desktop photos into actionable spatial metrics, visual heatmaps, and cleanup verification logs.

---

## Live Demo
**Live Web Application:** [https://deskzen-ubmk.onrender.com](https://deskzen-ubmk.onrender.com)

---

## Key Features

* **Instance Mask Segmentation:** Employs `YOLO11-Seg` to identify, outline, and isolate workspace items (input peripherals, cables, stationery, beverage containers, and electronics).
* **Cognitive Attention Heatmap:** Projects a 2D Gaussian friction field onto the primary reach zone (the lower 55% of the desk), visualizing high-distraction clutter clusters.
* **Calibrated Zen Score:** Calculates a dynamic workspace cleanliness index (0–100) anchored to true free surface area and weighted clutter penalties.
* **Dual-Pass Verification:** Allows users to scan their workspace "Before" and "After" cleaning, providing an interactive horizontal split-wipe slider and delta tracking (points gained, objects cleared, surface restored).
* **Actionable Punch List:** Automatically compiles categorized recommendations prioritized by ergonomic and safety impact (e.g., beverage spill risk near peripherals, cable tangle management).

---

## System Architecture

User Desk Photo
│
▼
Flask Backend (app.py)
│
├──> Ultralytics YOLO11-Seg (Bounding boxes + Polygon masks)
│         │
│         ├──> Pixel-Level Surface Availability Ratio
│         └──> Clutter Class Penalties
│
├──> 2D Gaussian Spatial Friction Field (OpenCV Turbo colormap)
│
└──> Physics-Calibrated Scoring Engine
│
▼
Clean JSON Telemetry + Base64 Rendered Overlays
│
▼
Interactive Glassmorphic UI (Before vs. After Comparison Slider)


---

## Tech Stack

* **Language:** Python 3.11
* **Backend:** Flask, Gunicorn
* **Computer Vision:** Ultralytics YOLO11-Seg, OpenCV (`opencv-python-headless`), NumPy
* **Deep Learning Framework:** PyTorch 2.5.1 (CPU optimized)
* **Frontend:** Vanilla HTML5, Modern CSS Glassmorphism, Responsive JavaScript
* **Cloud Infrastructure:** Render Web Services

---

## Local Development Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/venkatjyoshitpotnuru/DeskZen.git](https://github.com/venkatjyoshitpotnuru/DeskZen.git)
cd DeskZen

DeskZen/
├── .gitignore               # Excludes virtual environments and Python cache
├── .python-version          # Pins Python 3.11.9 for cloud deployment
├── README.md                # Project documentation
├── requirements.txt         # Production dependencies
├── app.py                   # Flask server, segmentation pipeline & scoring engine
├── yolo11n-seg.pt           # YOLO11-Seg neural network weights
└── templates/
    └── index.html           # Single-page UI with Before/After comparison slider
