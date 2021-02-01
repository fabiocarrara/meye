import argparse

import imageio
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from tqdm import tqdm
from utils import draw_predictions, compute_metrics


def main(args):
    video = imageio.get_reader(args.video)
    n_frames = video.count_frames()
    fps = video.get_meta_data()['fps']
    frame_w, frame_h = video.get_meta_data()['size']

    model = load_model(args.model)
    input_shape = model.input.shape[1:3]

    # default RoI
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

    out_video = imageio.get_writer(args.output_video, fps=fps)

    cropped = map(preprocess, video)
    frames_and_predictions = map(lambda x: (x, predict(x)), cropped)

    with open(args.output_csv, 'w') as out_csv:
        print('frame,pupil-area,pupil-x,pupil-y,eye,blink', file=out_csv)
        for idx, (frame, predictions) in enumerate(tqdm(frames_and_predictions, total=n_frames)):
            pupil_map, tags = predictions
            is_eye, is_blink = tags.squeeze()
            (pupil_y, pupil_x), pupil_area = compute_metrics(pupil_map, thr=args.thr, nms=True)

            row = [idx, pupil_area, pupil_x, pupil_y, is_eye, is_blink]
            row = ','.join(list(map(str, row)))
            print(row, file=out_csv)

            img = draw_predictions(frame, predictions, thr=args.thr)
            img = np.array(img)
            out_video.append_data(img)

    out_video.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('video', type=str, default='<video0>', help='Video file to process (use \'<video0>\' for webcam)')

    parser.add_argument('-t', '--thr', type=float, default=0.5, help='Map Threshold')
    parser.add_argument('-rl', type=int, help='RoI X coordinate of top left corner')
    parser.add_argument('-rt', type=int, help='RoI Y coordinate of top left corner')
    parser.add_argument('-rr', type=int, help='RoI X coordinate of right bottom corner')
    parser.add_argument('-rb', type=int, help='RoI Y coordinate of right bottom corner')

    parser.add_argument('-ov', '--output-video', default='predictions.mp4', help='Output video')
    parser.add_argument('-oc', '--output-csv', default='pupillometry.csv', help='Output CSV')

    args = parser.parse_args()
    main(args)
