# Configurator

`configurator` is a Python library module that provides a graphical interface for defining regions of interest (ROIs) directly on input tensors (e.g., images) using OpenCV. It allows users to draw and manipulate shapes using the mouse, making it easy to visually configure input data for further processing or analysis.

## ✨ Features

- Interactive OpenCV window for ROI drawing
- Draw **rectangles** and **circles** by left-clicking and dragging
- Define **polygons** using multiple left mouse clicks
- Middle-click (scroll wheel) to finalize and **store a shape**
- Right-click to **undo** the last point (for polygons) or remove the last shape
- Save all defined shapes automatically to `areas.json`


## Use Cases

- Preprocessing for computer vision applications
- ROI definition
- Interactive configuration for real-time applications
- Custom input region setup for deep learning pipelines

## 🖱 Mouse Controls

| Action                         | Behavior                                          |
|--------------------------------|---------------------------------------------------|
| Left-click + drag              | Draw a rectangle or circle                        |
| Multiple left-clicks           | Define points for a polygon                       |
| Scroll wheel click (middle)    | Finalize and add current shape to shape list      |
| Right-click                    | Undo last point (polygon) or remove last shape    |


## Installation

```bash
uv pip install -e tools/configurator
uv run configurator
```

## License
IMX500 Sample Applications is licensed under Apache License Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>