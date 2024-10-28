# Queue Monitoring

## Installation

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd [THIS_REPO]
$ pip install -e .
$ sudo apt-get install python3-pil python3-pil.imagetk
```
## Queue Monitor 
Queue Monitor. An edge application to track people in queues to help optimize queue management to reduce waiting times and enhance efficiency. Uses Object Detection and can be applied in retail, airports, and banks, and can also be used to analyze car traffic queues to help reduce traffic congestion in real time.
### Models
Model used in this example is an Nanodet Object Detection model to provide boundary boxes. To run the application you can get a quantized model using our [Model Compression Toolkit tutorial (MCT) Nanodet - Keras](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/imx500_notebooks/keras/example_keras_nanodet_plus_for_imx500.ipynb) 

Once the model has been created it is time to convert it and package it to be able to run on this platform. The [tutorial](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/develop/ai-tutorials/prepare-and-deploy-ai-models-tutorial?version=2024-09-27&progLang=) explains the process to convert ```model_name.keras``` to ```model_name.rpk```.

Or you can use a model already converted on [Rasberry Pi's model zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416_pp.rpk) 

### Application
Sample Application is configured to look at people, however to configure it to look at other object you can change the class ID or add multiple to detect multiple classes in a queue. Application needs a .json file to run where you store the x and y coords for the queue spaces. Format is shown in example.json provided.

#### Queue Monitor Args Options
```
--model                            Path of the model                                       [required]
--fps                              Maximum frames per second
--json-file                        Json file containing bboxes of queues                   [required]
--box-min-confidence               Confidence thershold for bounding box predictions
--iou-thereshold                   IoU thershold for Non-Maximum Suppressions (NMS)
--max-out-detection                Maximum number of output detections to keep after NMS
```
To run:

```
queue-monitor --model network-file-location.rpk --json-file example.json
```
:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.

To change the queue spaces, edit the example.json to add and edit the queue areas

## Issues
Knows issue of ImportError: cannot import name 'ImageTK'
To fix it "sudo apt-get install python3-pil python3-pil.imagetk"

