# MEYE pupillometry on MATLAB

> Try MEYE on a standalone [Web-App](https://www.pupillometry.it/)

> Learn more on the original [MEYE repo](https://github.com/fabiocarrara/meye)

> Label your own dataset with [pLabeler](https://github.com/LeonardoLupori/pLabeler)

Starting from MATLAB version 2021b, MEYE is also available for use on MATLAB!

Here's a brief tutorial on how to use it in you own experiments.

## What do you need?

- [MATLAB 2021b](https://it.mathworks.com/products/matlab.html) or later
- MATLAB Image Processing Toolbox
- [MATLAB Deep Learning Toolbox](https://it.mathworks.com/products/deep-learning.html)  
    An additional _support package_ of this toolbox has to be downloaded manually from the Add-On explorer in MATLAB:
    -  _Deep Learning Toolbox™ Converter for ONNX Model Format_
    ![image](https://user-images.githubusercontent.com/39329654/152327789-dde0af9b-d531-40be-b1a0-5ba17c508a13.png)
- A MEYE model in [ONNX](https://onnx.ai/) format. You can download out latest model [here](https://github.com/fabiocarrara/meye/releases).
![onnxModel](https://user-images.githubusercontent.com/39329654/152552616-1b800398-5794-4f51-b4ed-2e3339cb2d0d.png)


## Quick start!

```matlab
% Create an instance of Meye
meye = Meye('path/to/model.onnx');

% Example 1
% Make predictions on a single Image
%
% Load an image for which you want to predict the pupil
img = imread('path/to/img.tif');
% Make a prediction on a frame
[pupil, isEye, isBlink] = meye.predictImage(img);

% Example 2
% Make predictions on a video file and preview the results
%
meye.predictMovie_Preview('path/to/video');
```


# Known issues

## Upsample layers
When [importing](https://it.mathworks.com/help/deeplearning/ref/importonnxnetwork.html) a ONNX network, MATLAB tries to translate all the layers of the network from ONNX Operators to built-in MATLAB layers (see [here](https://it.mathworks.com/help/deeplearning/ref/importonnxnetwork.html#mw_dc6cd14c-e8d0-4370-af81-96626a888d9c)).  
This operation is not succesful for all the layers and MATLAB tries to overcome erros by automatically generating custom layers to replace the ones that it wasnt able to translate. These _custom_ layers are stored in a folder as MATLAB `.m` class files.  
We found a small bug in the way MATLAB translates `Upsample` layers while importing MEYE network. In particular, the custom generated layers perform the upsample with the `nearest` interpolation method, while it should be used the `linear` method for best results.  
For now, we solved this bug by automatically replacing the `nearest` method with the `linear` one in all the custom generated layers. This restores optimal performance with no additional computational costs, but it's a bit hacky.   
We hope that in future releases MATLAB's process of translation to its own built-in layers will be smoother and this trick will not be needed anymore.