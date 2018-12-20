import argparse

import cv2
import numpy as np
from keras.models import load_model

from tqdm import tqdm
from utils import visualizable, compute_metrics


def predictions(model, cap, process=None):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if process:
            frame = process(frame)
        frame = frame[:, :, None]  # same as np.expand_dims
        pred = model.predict(frame[None, ...])[0]
        yield frame, pred

    cap.release()


def main(args):

    # x_shape = (args.rh, args.rw, 1)
    # y_shape = (args.rh, args.rw, 2)

    # model = build_model(x_shape, y_shape)
    # model.load_weights('best_weights.hdf5')
    model = load_model('meye-segmentation-2018-12-20.hdf5')
    cap = cv2.VideoCapture(args.video)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # interval = 1000 / fps

    # y, x, h, w = (51, 70, 128, 128)  # RoI
    def roi(frame):
        eye = frame[args.ry:args.ry + args.rh, args.rx:args.rx + args.rw]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = eye.astype(np.float32) / 255.0
        return eye

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output, fourcc, fps, (args.rh, args.rw))

    for frame, pred in tqdm(predictions(model, cap, roi), total=n_frames):
        img = visualizable(frame, pred)
        img = (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB float -> BGR uint8
        out.write(img)

    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('video', type=str, help='Video file to process')
    parser.add_argument('-ry', type=int, default=0, help='RoI Y coordinate of top left corner')
    parser.add_argument('-rx', type=int, default=0, help='RoI X coordinate of top left corner')
    parser.add_argument('-rh', type=int, default=128, help='RoI height (-1 for full height)')
    parser.add_argument('-rw', type=int, default=128, help='RoI width (-1 for full width)')
    parser.add_argument('-o', '--output', default='predictions.mp4', help='Output video')

    args = parser.parse_args()
    main(args)
