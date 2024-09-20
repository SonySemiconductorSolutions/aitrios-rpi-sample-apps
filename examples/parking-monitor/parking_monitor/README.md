# Solutions

## Installation

Create a venv (virtual environment) with:
```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ cd [THIS_REPO]
$ apt-get install python3-pil python3-pil.imagetk
$ queue-monitor
```
## AI Gym 
Workout Monitoring, An edge application that tracks multiple people in real time with keypoints and bboxes to analyse the amount of reps they do in an exercise group. Providing feedback on the users workout to make better informed decisions and optimize performance during the workout to prevent injuries.
### Models
Model used is YOLOv8n_pose to provide both boundary boxes and keypoints of detections. Can be found "model zoo link"

Type of pose to detect ('pullup', pushup, abworkout, squat). To run:

```
python app_workout.py --model network/network-name.rpk
```

