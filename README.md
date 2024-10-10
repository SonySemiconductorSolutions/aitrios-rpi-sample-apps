# Introduction
Python based examples for the Raspberry PI AI Camera based on the IMX500 AI image sensor.

These examples are intended to be used in a virtual environment. The examples are based on the Picamera2 software.

## Installation
To list the available examples:
```
$ ls  examples/
```

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd examples/ # Press tab to get list of avaliable examples.
$ pip install -e .
```

After this step it is recommended to read the specific readme file from the examples folder if there are more steps needed or how to start the demo.

## Models

Most models in this repository are generic from the provided [model zoo](https://github.com/raspberrypi/imx500-models/tree/main).

Models used in the applications are: 
- **Highvis** - Uses a Custom trained Nanodet Model
- **Line Monitoring** - Uses a Custom Classification Model

To convert you own models follow our [MCT Tutorials](https://github.com/sony/model_optimization/tree/main/tutorials/notebooks/imx500_notebooks) to quantize your model. 
Then once  the model has been quantized it is time to convert it and package it to be able to run on this platform. The [tutorial](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/develop/ai-tutorials/prepare-and-deploy-ai-models-tutorial?version=2024-09-27&progLang=) explains the process to convert a ```model_name.keras``` or ```model_name.onnx``` to ```model_name.rpk```.

## Notice

### Security

Please read the Site Policy of GitHub and understand the usage conditions.
