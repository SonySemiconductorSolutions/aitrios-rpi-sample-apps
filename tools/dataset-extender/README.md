# Dataset augumentation script

## 💻 Installation

```
$ cd dataset-extender
$ uv run -m dataset-extender -i [PATH_TO_IMAGES]
```

Running this will create a default output folder called ```output```.

This folder will generate more images from the current provided folder that has already been divided in to subfolder representing images for each class.


### Example of creating a classification dataset for line-monitor

This is an example on how to create an augumented dataset for the line-monitor application that detects bolts that are with or without a nut.
From the raw dataset that contains a limited amount of images the dataset-extender will create more artificial manipulated images to simulate different light conditions, angles, sizes and image quality.


#### Prerequisite:
* The installation described at the top of this README.

#### Steps
Extract the images from line-monitor, the final files will be extracted to a folder called: ```training_dataset_nuts_and_bolts/```.
```
$ cd [THIS GIT]/tools
$ unzip ../examples/line-monitor/assets/training_dataset_nuts_and_bolts.zip -d .
```

Running dataset-extender with the -h switch will list the available options. In this example we will just us the default settings and only specifying the input data folder.
```
$ uv run -m dataset-extender -h
$ uv run -m  dataset-extender -i training_dataset_nuts_and_bolts/
```
The final result will be placed in the folder ```output```.
The folders ```train```, ```val``` and ```test``` can be used as training data for [Brain Builder](https://developer.aitrios.sony-semicon.com/en/studio/brain-builder) where we train and quantize the model easily.


