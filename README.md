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
