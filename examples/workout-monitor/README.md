# Workout Monitoring

## Installation

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd [THIS_REPO]
$ pip install -e .
```
## AI Gym 
Workout Monitoring, An edge application that tracks people in real time with keypoints and bboxes to analyse the amount of reps they do in an exercise group. Providing feedback on the users workout to make better informed decisions and optimize performance during the workout to prevent injuries. 
### Models
Model used in this example is Pose Model to provide both boundary boxes and keypoints to the application.  To run the application you can get a converted model on Raspberry Pi models [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk)

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
:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.
