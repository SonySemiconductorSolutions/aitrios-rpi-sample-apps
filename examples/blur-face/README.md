<div align="center">

<img src="assets/blur-face.gif" alt="Alt Text" width="400" height="300">


</div>

<div align="center">

# Blur Face

</div>

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)


The Blur Face application is designed to enhance privacy by blurring the faces of individuals detected in images or video frames. By using advanced facial detection technology, it accurately identifies faces and applies a blurring effect to obscure them. This ensures that personal identities are protected, making the application an invaluable tool for safeguarding privacy in contexts such as social media, public surveillance, and the handling of sensitive data.


## 🚀 Installation and Start

Before running the blur face application, make sure you are inside this application's directory.

Then using uv to run the application, which will install the pyproject.toml and start the application:
```bash
uv run app.py
```

### 🧠 Models Used
Model used in this example is Posenet Model to get the keypoints of the face. You can get a converted model on Raspberry Pi models [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_posenet.rpk). 

:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.

## License
IMX500 Sample Applications is licensed under Apache License Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>
