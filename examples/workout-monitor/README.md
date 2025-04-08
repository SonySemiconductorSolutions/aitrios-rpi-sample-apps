<div align="center">

<img src="assets/pull_ups.png" alt="Alt Text" width="400" height="300">


</div>

<div align="center">

# Workout Monitor

</div>

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)


Workout Monitoring, An edge application that tracks people in real time with keypoints and bboxes to analyse the amount of reps they do in an exercise group. Providing feedback on the users workout to make better informed decisions and optimize performance during the workout to prevent injuries. 

Type of workout poses to detect **pullup, pushup, abworkout, squat** and can change between the different workout types when starting the application.  

## 🚀 Installation and Start

Before running the workout-monitor application, create a virtual environment inside the application's directory with uv.
```
# create virtual environment using uv
$ uv venv --system-site-packages
```

Then run the application:
```
# Installs the pyproject.toml settings and starts the app
$ uv run app.py --exercise squat
```

### 🧠 Models Used
Model used in this example is Higherhrnet Model to provide both boundary boxes and keypoints to the application. You can get a converted model on Raspberry Pi models [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk) for Pose Estimation.

### 📝 Args Options
```
--exercise              Type of exercise to monitor            options: [pullup, pushup, abworkout, squat]      
```

:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.
