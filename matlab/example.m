%% Download all the example material
% 
% 1 - Download the latest MEYE model in ONNX format
% -------------------------------------------------------------------------
% Download the .onnx file from the assets here:
% https://github.com/fabiocarrara/meye/releases

% EXAMPLE data can be found in this folder:
% https://drive.google.com/drive/folders/1BG6O5BEkwXkNKC_1XuB3H9wbx3DeNWwF?usp=sharing
% 
% 2 - Download an example image of a simple mouse eye from:
% https://drive.google.com/file/d/1hcWcC1cAmzY4r-SIWDIgUY0-gpbmetUL/view?usp=sharing
% 
% 3 - Download an example of a large image here:
% https://drive.google.com/file/d/16QixvUMtojqfrcy4WXlYJ7CP3K8vrz_C/view?usp=sharing
% 
% 4 - Download an example pupillometry video here:
% https://drive.google.com/file/d/1TYj80dzIR1ZjpEvfefH_akhbUjwpvJta/view?usp=sharing


%% EXAMPLE 1
% -------------------------------------------------------------------------
% Predict the pupil from a simple image of an eye

% Clean up the workspace
clearvars, clc

% Change these values according to the filenames of the MEYE model and the
% simple pupil image
MODEL_NAME = 'meye_20220124.onnx';
IMAGE_NAME = 'pupilImage_simple.png';


% Initialize a MEYE object
meye = Meye(MODEL_NAME);

% Load the simple image
img = imread(IMAGE_NAME);

% Predict a single image
[pupilMask, eyeProb, blinkProb] = meye.predictImage(img);

% Plot the results of the prediction
subplot(1,3,1)
imshow(img)
title('Original Image')

subplot(1,3,2)
imagesc(pupilMask)
title(sprintf('Prediction (Eye:%.2f%% - Blink:%.2f%%)',eyeProb*100,blinkProb*100))
axis off, axis image

subplot(1,3,3)
imshowpair(img, pupilMask)
title('Merge')


%% EXAMPLE 2
% -------------------------------------------------------------------------
% Binarize the pupil prediction and get the pupil size in pixels

% Clean up the workspace
clearvars, close all,  clc

% Change these values according to the filenames of the MEYE model and the
% simple pupil image
MODEL_NAME = 'meye_20220124.onnx';
IMAGE_NAME = 'pupilImage_simple.png';


% Initialize a MEYE object
meye = Meye(MODEL_NAME);

% Load the simple image
img = imread(IMAGE_NAME);

% Predict a single image
% You can automatically binarize the prediction by passing the "threshold"
% optional argument. This number can be between 0 and 1. If omitted, the
% function returns a raw probability map instead of a binarized image
pupilBinaryMask = meye.predictImage(img, 'threshold', 0.4);

imshowpair(img, pupilBinaryMask)
title(sprintf('Pupil Size: %u px', sum(pupilBinaryMask,'all')))


%% EXAMPLE 3
% -------------------------------------------------------------------------
% Predict the pupil on a large image where the eye is a small portion of
% the image

% Clean up the workspace
clearvars, close all,  clc

% Change these values according to the filenames of the MEYE model and the
% simple pupil image
MODEL_NAME = 'meye_20220124.onnx';
IMAGE_NAME = 'pupilImage_large.png';


% Initialize a MEYE object
meye = Meye(MODEL_NAME);

% Load the simple image
img = imread(IMAGE_NAME);

% Predict the image
pupilMask = meye.predictImage(img);

% As you can see from this image, the prediction is not perfect. This is
% because MEYE was trained on images that tightly contained the eye. 
subplot(1,2,1)
imshowpair(img, pupilMask)
title('Tomal Image prediction (low-quality)')

% In order to solve this issue it is possible to restrict the prediction to
% a rectangular Region of Interest (ROI) in the image. This is done simply
% by passing the optional argument "roiPos" to the predictImage function.
% The roiPos is a 4-elements vector containing X,Y, width, height of a
% rectangular shape. Note that X and Y are the coordinates of the top left
% corner of the ROI

ROI = [90,90,200,200];
pupilMask = meye.predictImage(img, 'roiPos', ROI);

% Plot the results with the ROI and see the difference between the 2 methods
subplot(1,2,2)
imshowpair(img, pupilMask)
rectangle('Position',ROI, 'LineStyle','-.','EdgeColor',[1,0,0])
title('ROI prediction (high quality)')
linkaxes
set(gcf,'Position',[300,600,1000,320])


%% EXAMPLE 4
% -------------------------------------------------------------------------
% Show a preview of the prediction of an entire pupillometry video.
% 
% As you saw you can adjust a few parameters for the prediction.
% If you want to get a quick preview of how your pre-recorded video will be
% processed, you can use the method predictMovie_Preview.
% Here you can play around with different ROI positions and threshold
% values and see what are the results before analyzing the whole video.

% Clean up the workspace
clearvars, close all,  clc

% Change these values according to the filenames of the MEYE model and the
% simple pupil image
MODEL_NAME = 'meye_20220124.onnx';
VIDEO_NAME = 'mouse_example.mp4';

% Initialize a MEYE object
meye = Meye(MODEL_NAME);

% Try to play around moving or resizing the ROI to see how the performances change
ROI = [70, 60, 200, 200]; 

% Change the threshold value to binarize the pupil prediction.
% Use [] to see the raw probability map. Use a number in the range [0:1] to binarize it
threshold = 0.4;            

meye.predictMovie_Preview(VIDEO_NAME,"roiPos", ROI,"threshold",threshold);



%% EXAMPLE 5
% Predict the entire video and get the results table

% Clean up the workspace
clearvars, close all,  clc

% Change these values according to the filenames of the MEYE model and the
% simple pupil image
MODEL_NAME = 'meye_20220124.onnx';
VIDEO_NAME = 'mouse_example.mp4';

% Initialize a MEYE object
meye = Meye(MODEL_NAME);

% Try to play around moving or resizing the ROI to see how the performances change
ROI = [70, 60, 200, 200]; 

% Change the threshold value to binarize the pupil prediction.
% Use [] to see the raw probability map. Use a number in the range [0:1] to binarize it
threshold = 0.4;

% Predict the whole movie and save results in a table
T = meye.predictMovie(VIDEO_NAME, "roiPos", ROI, "threshold", threshold);

% Show some of the values in the table
disp(head(T))

% Plot some of the results
subplot 311
plot(T.frameTime,T.isEye, 'LineWidth', 2)
title('Eye Probability')
ylabel('Probability'),
xlim([T.frameTime(1) T.frameTime(end)])

subplot 312
plot(T.frameTime,T.isBlink, 'LineWidth', 2)
title('Blink Probability')
ylabel('Probability')
xlim([T.frameTime(1) T.frameTime(end)])

subplot 313
plot(T.frameTime,T.pupilArea, 'LineWidth', 2)
title('Pupil Size')
xlabel('Time (s)'), ylabel('Pupil Area (px)')
xlim([T.frameTime(1) T.frameTime(end)])
