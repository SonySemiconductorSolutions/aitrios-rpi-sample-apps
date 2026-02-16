<div align="center">

<img src="assets/highvis.gif" alt="Alt Text" width="400" height="300">


</div>

<div align="center">

# Highvis

</div>

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)


Matching Overlapping Objects with Robust Filtering.

This demo showcases a system designed to match overlapping objects detected by object detection AI model. It includes filtering mechanisms to ensure objects are not lost even if the AI model struggles detecting the objects over a series of continuous frames.
In this particular case the AI model has been trained to detect people and vests.

## Creating a custom NanoDet model to run this example.

To run this example a custom nanodet object detection model has to be trained. The dataset used is the PPE v.3 from RoboFlow.
This link to the tutorial that explains the procedure to train a [nanodet object detector](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training/blob/main/notebooks/nanodet-ppe/custom_nanodet.ipynb).
Once the you have a ```packerOut.zip``` and ```labels.txt```  it is time to package it to be able to run on this platform. The [tutorial](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/develop/ai-tutorials/prepare-and-deploy-ai-models-tutorial) explains the process to convert a model. However to convert a model quickly, we will use modlib and upload the model.

> [!NOTE] 
> If you wish to use a different model architecture you may have to use a different post processor and color_format. Check out the [post processors](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/blob/main/modlib/models/post_processors/post_processors.py) in Modlib to see what one to use for your model 

## 🚀 Installation and Start

Before running the highvis application, in this application's directory you need to create a network folder where the model file and labels.txt are located and can be accessed easily.

```
$ mkdir network
$ cp -v [MODEL_PATH]/packerOut.zip highvis/network
$ cp -v [LABELS_PATH]/labels.txt highvis/network
```

Then create a virtual environment with uv and run the application:
```
uv run app.py --model network/packerOut.zip
```

Or if you wish to save the results to save the output to json file to continue development further:

```
uv run app-json.py --model network/packerOut.zip
```

### 🧠 Model Used

- **NanoDet Model**:
  - NanoDet is a FCOS-style one-stage anchor-free object detection model which using Generalized Focal Loss as classification and regression loss.

#### 📝 Highvis Args Options

`--model <path>`     _(required)_ : Path to custom trained highvis model, must be packerOut.zip file

:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.

## 🏛️ Architecture Overview

### Bottom-Up Approach

1. **AI Model (Object Detection)**:
   - Utilizes object detection models such as NanoDet to identify objects.

2. **Output Tensor Transformation**:
   - Converts the output tensors from the AI model to a format which is suitable for the tracker.

3. **Tracker**:
   - Assigns a unique tracker ID to each detected object.
   - Maintains these IDs consistently over time and across frames.

4. **Matcher**:
   - Calculates and identifies which objects are overlapping, the frame ID from the tracker is kept in the cache.

5. **Business Logic Layer**:
   - Computes statistics and provides information about the current state.
   - Tracks the number of people with and without high-visibility vests.


This architecture ensures robust tracking and accurate object matching even in challenging scenarios where objects may overlap or be intermittently lost by the detection model. The combination of advanced detection models, effective tracking, and intelligent filtering makes this system reliable for real-world applications.

## License
IMX500 Sample Applications is licensed under Apache License Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>