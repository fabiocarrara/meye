FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN apt update && apt install -y nvidia-modprobe
RUN pip install adabelief-tf pandas imageio imageio-ffmpeg seaborn sklearn tqdm
RUN pip install tensorflowjs

