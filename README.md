# [*mEye*](https://www.pupillometry.it): A Deep Learning Tool for Pupillometry

> ⭐ MEYE is available on **MATLAB**! Check it out [here](matlab/README.md)

> Check out [pupillometry.it](https://www.pupillometry.it) for a ready-to-use web-based mEye pupillometry tool!

This branch provides the Python code to make predictions and train/finetune models.
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

  1. Download the [pretrained model](https://github.com/fabiocarrara/meye/releases/download/v0.1/meye-segmentation_i128_s4_c1_f16_g1_a-relu.hdf5).
  2. Check out the `pupillometry-offline-videos.ipynb` notebook for a complete example of pupillometry data analysis.
  3. In alternative, we provide also the `predict.py` script that implements the basic loop to make predictions on video streams. E.g.:

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

  1. Download our dataset ([NN_human_mouse_eyes.zip](https://doi.org/10.5281/zenodo.4488164), 246.4 MB) or prepare your dataset following our dataset's structure.
     > If you need to annotate your dataset, check out [pLabeler](https://github.com/LeonardoLupori/pLabeler), a MATLAB software for labeling pupil images.
     
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

## MATLAB support
Starting from MATLAB version 2021b, MEYE is also available for use on MATLAB!  
A fully functional class and a tutorial for its use is available [here](matlab/README.md)!


## References

### Dataset
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4488164.svg)](https://doi.org/10.5281/zenodo.4488164)

If you use our dataset, please cite:

     @dataset{raffaele_mazziotti_2021_4488164,
       author       = {Raffaele Mazziotti and Fabio Carrara and Aurelia Viglione and Lupori Leonardo and Lo Verde Luca and Benedetto Alessandro and Ricci Giulia and Sagona Giulia and Amato Giuseppe and Pizzorusso Tommaso},
       title        = {{Human and Mouse Eyes for Pupil Semantic Segmentation}},
       month        = feb,
       year         = 2021,
       publisher    = {Zenodo},
       version      = {1.0},
       doi          = {10.5281/zenodo.4488164},
       url          = {https://doi.org/10.5281/zenodo.4488164}
     }
