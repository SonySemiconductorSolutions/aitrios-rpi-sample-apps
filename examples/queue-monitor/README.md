# Solutions - Queue Monitoring

## Installation

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd [THIS_REPO]
$ pip install -e .
$ apt-get install python3-imaging python3-pil.imagetk
```
## Queue Management 
Queue Management. An edge application to track people in queues to help optimize queue management to reduce waiting times and enhance efficiency. Uses YOLOv8n and can be applied in retail, airports, and banks, and can also be used to analyze car traffic queues to help reduce traffic congestion in real time.
### Models
Model used in this example is YOLOv8n to provide boundary boxes but any Object detection model can be used. To train a your're own YOLOv8n model please check our [Model Compression Toolkit tutorial (MCT) - Yolov8n Keras](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/imx500_notebooks/keras/keras_yolov8n_for_imx500.ipynb) or [Model Compression Toolkit tutorial (MCT) - Yolov8n PyTorch](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/imx500_notebooks/pytorch/pytorch_yolov8n_for_imx500.ipynb)

### Application
Sample Application is configured to look at people, however to configure it to look at other object you can change the class ID or add multiple to detect multiple classes in a queue. Application also includes a point selector application to run where you can take live image of camera and darw the queues on your image and save it to a json file 

#### Point Selector Args Options
```
--filename                        Path of the json file                                    [required]
```

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
pts-selector --filename example.json
queue-monitor --model network-file-location.rpk --json-file example.json
```

## Issues
Knows issue of ImportError: cannot import name 'ImageTK'
To fix it "apt-get install python3-imaging python3-pil.imagetk"

