# Parking-Monitor

## Installation

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd [THIS_REPO]
$ pip install -e .
$ sudo apt-get install python3-pil python3-pil.imagetk
```
## Parking-Monitor
**AI-Parking-Monitor** is an application designed to monitor a parking lot using the Raspberry pi AI-camera system. The application provides real-time data on the availability of parking spaces, displaying the status of each space visually and in a machine-readable JSON format. This project is useful for managing parking spaces, ensuring efficient space utilization, and providing an overview of parking availability.
### Models
Model used in this example is an Nanodet Object Detection model to provide boundary boxes. To run the application you can get a quantized model using our [Model Compression Toolkit tutorial (MCT) Nanodet - Keras](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/imx500_notebooks/keras/example_keras_nanodet_plus_for_imx500.ipynb)

Once the model has been created it is time to convert it and package it to be able to run on this platform. The [tutorial](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/develop/ai-tutorials/prepare-and-deploy-ai-models-tutorial?version=2024-09-27&progLang=) explains the process to convert ```model_name.keras``` to ```model_name.rpk```.

Or you can use a model already converted on [Rasberry Pi's model zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416_pp.rpk) 

### Application
Sample Application is configured to look at vehicles, however to configure it to look at other object you can change the class ID or add multiple to detect multiple classes in a queue. Application needs a .json file to run where you store the x and y coords for the parking spaces. Format is shown in config.json provided.

## Features

- **Real-Time Monitoring**: Continuously analyzes the parking lot to detect occupied and free parking spaces.
- **Visual Feedback**: Displays a live feed with bounding boxes:
  - **Green**: Free parking space.
  - **Red**: Occupied parking space.
- **Scalable**: Can be adapted for parking lots of various sizes.

#### Parking-Monitor Args Options
```
--model                            Path of the model                                       [required]
--fps                              Maximum frames per second
--json-file                        Json file containing bboxes of parking spaces                   [required]
--box-min-confidence               Confidence thershold for bounding box predictions
--iou-thereshold                   IoU thershold for Non-Maximum Suppressions (NMS)
--max-out-detection                Maximum number of output detections to keep after NMS
```
To run:

```
parking-monitor --model network-file-location.rpk --json-file example.json
```
:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.

To change the parking spaces, edit the config.json to add and edit the point areas

## Issues
Knows issue of ImportError: cannot import name 'ImageTK'
To fix it "sudo apt-get install python3-pil python3-pil.imagetk"

