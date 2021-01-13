import argparse

import imageio
import numpy as np

from keras.models import load_model
from PIL import Image, ImageOps
from tqdm import tqdm
from utils import draw_predictions, compute_metrics


def main(args):
    video = imageio.get_reader(args.video)
    n_frames = video.count_frames()
    fps = video.get_meta_data()['fps']
    frame_w, frame_h = video.get_meta_data()['size']

    # model = build_model(x_shape, y_shape)
    # model.load_weights('best_weights.hdf5')
    model = load_model(args.model)
    input_shape = model.input.shape[1:3]

    # y, x, h, w = (51, 70, 128, 128)  # RoI
    if None in (args.rl, args.rt, args.rr, args.rb):
        side = min(frame_w, frame_h)
        args.rl = (frame_w - side) / 2
        args.rt = (frame_h - side) / 2
        args.rr = (frame_w + side) / 2
        args.rb = (frame_h + side) / 2

    crop = (args.rl, args.rt, args.rr, args.rb)

    def preprocess(frame):
        frame = Image.fromarray(frame)
        eye = frame.crop(crop)
        eye = ImageOps.grayscale(eye)
        eye = eye.resize(input_shape)
        return eye

    def predict(eye):
        eye = np.array(eye).astype(np.float32) / 255.0
        eye = eye[None, :, :, None]
        return model.predict(eye)

    out = imageio.get_writer(args.output, fps=fps)

    cropped = map(preprocess, video)
    frames_and_predictions = map(lambda x: (x, predict(x)), cropped)

    for frame, predictions in tqdm(frames_and_predictions, total=n_frames):
        img = draw_predictions(frame, predictions, thr=args.thr)
        img = np.array(img)
        out.append_data(img)

    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('video', type=str, help='Video file to process')

    parser.add_argument('-t', '--thr', type=float, default=0.5, help='Map Threshold')
    parser.add_argument('-rl', type=int, help='RoI X coordinate of top left corner')
    parser.add_argument('-rt', type=int, help='RoI Y coordinate of top left corner')
    parser.add_argument('-rr', type=int, help='RoI X coordinate of right bottom corner')
    parser.add_argument('-rb', type=int, help='RoI Y coordinate of right bottom corner')

    parser.add_argument('-o', '--output', default='predictions.mp4', help='Output video')

    args = parser.parse_args()
    main(args)
