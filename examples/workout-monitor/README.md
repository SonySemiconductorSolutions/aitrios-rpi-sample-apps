# Solutions - Workout Monitoring

## Installation

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd [THIS_REPO]
$ pip install -e .
```
## AI Gym 
Workout Monitoring, An edge application that tracks multiple people in real time with keypoints and bboxes to analyse the amount of reps they do in an exercise group. Providing feedback on the users workout to make better informed decisions and optimize performance during the workout to prevent injuries. 
### Models
Model used in this example is YOLOv8n_pose to provide both boundary boxes and keypoints of detections. The pre trained model can be found on the Raspberry Pi file system ```/usr/share/imx500-models/imx500_network_yolov8n_pose.rpk```. To train a custom posenet model please check the [Model Compression Toolkit tutorials (MCT)](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/imx500_notebooks/pytorch/pytorch_yolov8n_pose_for_imx500.ipynb)

### Application
Type of workout poses to detect ('pullup', 'pushup', 'abworkout', 'squat') and can change between the different workout types when creating the gym object.  

#### Args Options
```
--model                            Path of the model                                       [required]
--fps                              Maximum frames per second
--box-min-confidence               Confidence thershold for bounding box predictions
--keypoint-min-confidence          Confidence thershold for keypoint predictions
--iou-thereshold                   IoU thershold for Non-Maximum Suppressions (NMS)
--max-out-detection                Maximum number of output detections to keep after NMS
--exercise 			   Type of exercise to monitor, default is pushup
```
To run:

```
workout-monitor --model network-file-location.rpk 
```


