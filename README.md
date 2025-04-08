<div align="center">
  <p>
    <a align="center">
      <img
        width="100%"
        src="assets/Sample_Apps_Banner.png"
      >
    </a>
  </p>

</div>

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)

# Introduction 👋
Here we create sample applications using the tools we produce to help the commuity learn and explore how easy it is to build AI applications around the IMX500 sensor. Where we encourage you to build your own applications and the ideas you have. All Sample Applications are Python based examples built to use the Raspberry PI AI Camera, based on the IMX500 AI image sensor.

# Install 🏗️
Clone the repo and run applications in their directories. All applications are intended to be used in a uv environment and have a pyproject.toml file to install requirments. To install uv:

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```
For more information on uv you can read their [documentation](https://docs.astral.sh/uv/getting-started/installation/)

## Application Module Library 
Application Module Library is a Python library that simplifies the development of end-to-end appllications for the IMX500 vision sensor. With seamless integration of AITRIOS tools, it helps developers streamline their workflow and focus on what matters most. You can find [Application Module Library Github page](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library) where you find more documentation on how to build your own applications using the library.

# Sample Applications 💻

<div align="center">
<p align="center">

  Sample Application  | Description | AI Model Type | Model Used 
-------------------- | -----------|--------------------|---------
[Highvis](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sample-apps/tree/main/examples/highvis) | Detects people and matches them to be wearing safety equipment (PPE). | Object Detection | [Custom NanoDet](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training/blob/main/notebooks/nanodet-ppe/custom_nanodet.ipynb) 
[Line Monitoring](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sample-apps/tree/main/examples/line-monitor) | Transforms production line monitoring and object classification, providing a smarter, more efficient way to ensure quality and operational excellence | Classification | [Brain Builder](https://developer.aitrios.sony-semicon.com/en/posts/quick-walkthrough-of-classifier-task-with-brain-builder/) | 
[Parking Monitoring](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sample-apps/tree/main/examples/parking-monitor) | Provides real-time data on the availability of parking spaces, displaying the status of each space visually  | Object Detection | [NanoDet Zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416_pp.rpk) 
[Queue Monitoring](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sample-apps/tree/main/examples/queue-monitor) |  Track people in queues to help optimize queue management to reduce waiting times and enhance efficiency. | Object Detection | [NanoDet Zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416_pp.rpk) 
[Workout Monitoring](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sample-apps/tree/main/examples/workout-monitor) |  Tracks people in real time with keypoints and bboxes to analyse the amount of reps they do in an exercise group. | Pose Estimation | [HigherHRNet](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk) 
</p>    
</div>

If you wish to use a Yolo model you can get pretrained models and train your own with [Ultralytics](https://docs.ultralytics.com/integrations/sony-imx500/) and convert them for IMX500 and use in our Sample Applications or your own Applications.

<img src="assets/Applications.png" alt="Alt Text">

## License
IMX500 Sample Applications is licensed under Apache License Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>