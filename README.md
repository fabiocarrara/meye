# [*mEye*](https://www.pupillometry.it): A Deep Learning Tool for Pupillometry

> Check out [pupillometry.it](https://www.pupillometry.it) for a ready-to-use web-based mEye pupillometry tool!

This branch provides the Python code for make predictions and train/finetune models.
If you are interested in the code of the pupillometry web app, check out the `gh-pages` branch.

## Requirements
You need a Python 3 environment with the following packages installed:

  - tensorflow >= 2.4
  - imageio, imageio-ffmpeg
  - scipy  
  - tqdm

If you want to train models, you also need 

  - adabelief_tf >= 0.2.1
  - pandas
  - sklearn

We provide a [Dockerfile](./Dockerfile) for building an image with docker.

## Make Predictions with Pretrained Models

You can make predictions with pretrained models on pre-recorded videos or webcam streams. 

  1. Download the pretrained model (URLs coming soon)
  2. Check out the `predict.py`. It implements the basic loop to make predictions on video stream. E.g.:

       - ```bash
         # input: webcam (default)
         # prediction roi: biggest central square crop (default)
         # outputs: predictions.mp4, predictions.csv (default)
         predict.py path/to/model
         ```
     
       - ```bash
         # input: video file
         # prediction roi: left=80, top=80, right=208, bottom=208
         # outputs: video_with_predictions.mp4, pupil_metrics.csv
         predict.py path/to/model path/to/video.mp4 -rl 80 -rt 80 -rr 208 -rb 208 -ov video_with_predictions.mp4 -oc pupil_metrics.csv
         ```
       - ```bash
         # check all parameters with
         predict.py -h
         ```
    
## Training Models

  1. Download our dataset (URL coming soon) or prepare your dataset following our dataset's structure.
     The dataset should be placed in `data/<dataset_name>`.
     
  2. If you are using a custom dataset, edit `train.py` to perform the train/validation/test split of your data.
     
  3. Train with default parameters:
     ```bash
     python train.py -d data/<dataset_name>
     ```
     
  - For a list of available parameters, run
    ```bash
    python train.py -h
    ```