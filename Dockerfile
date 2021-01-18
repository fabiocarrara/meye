FROM  tensorflow/tensorflow:1.8.0-gpu-py3

RUN apt update && apt install -y python3-tk ffmpeg
RUN pip install \
    keras==2.2.4 \
    matplotlib \
    imageio imageio-ffmpeg \
    tqdm

