# Demo: High-visibility vest usage detector

Matching Overlapping Objects with Robust Filtering.

This demo showcases a system designed to match overlapping objects detected by object detection AI model. It includes filtering mechanisms to ensure objects are not lost even if the AI model struggles detecting the objects over a series of continous frames.
In this particual case the AI model has been trained to detect people and vests.

## Creating a custom NanoDet model to run this example.

To run this example a custom nanodet object detection model has to be trained. The dataset used is the PPE v.3 from RoboFlow.
This link to the tutorial that explains the precedure to train a [nanodet object detector](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training-dev/blob/master/notebooks/nanodet-ppe/custom_nanodet.ipynb).
Once the model ```nanodet-quant-ppe.keras``` has been created it is time to convert it and package it to be able to run on this platform. TODO The [tutorial](https://draft-aitrios-portal.wpp.developer.sony.com/en/raspberrypi-ai-camera/tutorials/prepare-and-deploy-ai-models?version=2024-09-05&progLang=#_for_your_raspberry_pi) exlains the process to convert ```nanodet-quant-ppe.keras``` to ```network.rpk```.

 Copy the model to ```networks``` folder in this examples root folder.
 ```
 cp -v [PATH_TO_OUT]/network.rpk  ./networks/imx500_network_nanodet_ppe.rpk
 ```
 
## Installation and start

```
$ python3 -m venv venv --system-site-packages
$ source venv/bin/activate
$ pip install -e [PATH_TO_THIS_REPO]
```

Start app:
```
$ highvis
```

:warning: **Running a new example with new model for the first time can take a few minutes for the new model to be uploaded.

## Architecture Overview

### Bottom-Up Approach

1. **AI Model (Object Detection)**:
   - Utilizes object detection models such as SSD MobileNet V1 and YOLO V8 to identify objects with.

2. **Output Tensor Transformation**:
   - Converts the output tensors from the AI model to a format suitable for the tracker.

3. **Tracker**:
   - Assigns a unique tracker ID to each detected object.
   - Maintains these IDs consistently over time and across frames.

4. **Object Overlap Detector**:
   - Calculates and identifies which objects are overlapping, the frame ID from the tracker is kept in the cache.

5. **Business Logic Layer**:
   - Computes statistics and provides information about the current state.
   - Tracks the number of people with and without high-visibility vests.

### Models Used

- **SSD MobileNet V1**:
  - A Single Shot MultiBox Detector model using MobileNet architecture, optimized for mobile and real-time applications.
  

This architecture ensures robust tracking and accurate object matching even in challenging scenarios where objects may overlap or be intermittently lost by the detection model. The combination of advanced detection models, effective tracking, and intelligent filtering makes this system reliable for real-world applications.


## Changing Settings

The settings for this system are configured in the `settings.json` file with a default setup. This ensures that the system runs correctly with the selected model. This file will be generated at the first run of this application.


