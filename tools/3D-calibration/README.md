<div align="center">

# 3D calibration

</div>


[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)

Some applications require 3D measurements to calculate real world values from pixel values. This tool provides the means of calibrating your camera/scene to capture precise real world values. From this tool we get distance per pixel values to apply depth and real world measurements to the application. 

## 🚀 Installation and Start

> [!IMPORTANT] 
> #### 3D Calibration
> To launch 3D_calibration:
>```
>$ python3 3D_calibration.py --filename example.json
>```
> To use, click the take image button and start drawing the areas you wish to draw by clicking on image, 2 clicks draws a bbox area. Then click twice on an object of known size in the marked area, where the two points of the clicks are the size of the object. Then for the number of areas you have drawn, enter the size of the marked objects (in meters) in the entry box with spaces to separate values if there are multiple areas. For example, I marked an area where cars will be traveling through. I want to calculate the speed of the cars. To calibrate the scene, I mark the area I want then mark the width of the road, which is known to be 3.2 meters. 
>
> To calibrate multiple areas, draw your Areas you wish to calibrate. Then enter a real world measurements for each Area. For example, in my text box I might enter ```0.5 0.5 0.5``` for three Areas with a space between the 3 values.
>
>Requirements to run:
>```
>sudo apt-get install python3-pil python3-pil.imagetk
>```


#### 📝 3D_calibration Args Options
```
--filename               Json file name. Must end with .json                  [required]
```
## License
IMX500 Sample Applications is licensed under Apache License Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" /></a>
