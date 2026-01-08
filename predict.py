import argparse
import cv2
import imageio
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from tqdm import tqdm
from utils import draw_predictions, compute_metrics


def main(args):
    # Load the model
    model = load_model(args.model, compile=False)
    input_shape = model.input.shape[1:3]

    # Open video capture
    if args.video.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.video}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Backend: {cv2.getBackendName()}")
        
        # Try listing available cameras
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        print(f"Available camera indices: {arr}")
        
        raise IOError("Cannot open video source")

    # Get video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up RoI
    if None in (args.rl, args.rt, args.rr, args.rb):
        side = min(frame_w, frame_h)
        args.rl = int((frame_w - side) / 2)
        args.rt = int((frame_h - side) / 2)
        args.rr = int((frame_w + side) / 2)
        args.rb = int((frame_h + side) / 2)

    crop = (args.rl, args.rt, args.rr, args.rb)

    def preprocess(frame):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        eye = frame.crop(crop)
        eye = ImageOps.grayscale(eye)
        eye = eye.resize(input_shape)
        return eye

    def predict(eye):
        eye = np.array(eye).astype(np.float32) / 255.0
        eye = eye[None, :, :, None]
        return model.predict(eye, verbose=0)

    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_w, frame_h))

    with open(args.output_csv, 'w') as out_csv:
        print('frame,pupil-area,pupil-x,pupil-y,eye,blink', file=out_csv)
        
        pbar = tqdm(total=n_frames if n_frames > 0 else None)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            eye = preprocess(frame)
            predictions = predict(eye)
            pupil_map, tags = predictions
            is_eye, is_blink = tags.squeeze()
            (pupil_y, pupil_x), pupil_area = compute_metrics(pupil_map, thr=args.thr, nms=True)

            row = [frame_idx, pupil_area, pupil_x, pupil_y, is_eye, is_blink]
            row = ','.join(map(str, row))
            print(row, file=out_csv)

            img = draw_predictions(eye, predictions, thr=args.thr)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Draw ROI on the original frame
            cv2.rectangle(frame, (args.rl, args.rt), (args.rr, args.rb), (0, 255, 0), 2)
            
            # Resize the prediction image to match the ROI size
            img_resized = cv2.resize(img, (args.rr - args.rl, args.rb - args.rt))
            
            # Overlay the prediction on the original frame
            frame[args.rt:args.rb, args.rl:args.rr] = img_resized
            
            # Display the frame
            cv2.imshow('MEye Pupillometry', frame)
            
            out_video.write(frame)

            frame_idx += 1
            pbar.update(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.close()

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('model', type=str, help='Path to model')
    parser.add_argument('video', type=str, default='<video0>', help='Video file to process (use "webcam" for webcam)')

    parser.add_argument('-t', '--thr', type=float, default=0.5, help='Map Threshold')
    parser.add_argument('-rl', type=int, help='RoI X coordinate of top left corner')
    parser.add_argument('-rt', type=int, help='RoI Y coordinate of top left corner')
    parser.add_argument('-rr', type=int, help='RoI X coordinate of right bottom corner')
    parser.add_argument('-rb', type=int, help='RoI Y coordinate of right bottom corner')

    parser.add_argument('-ov', '--output-video', default='predictions.mp4', help='Output video')
    parser.add_argument('-oc', '--output-csv', default='pupillometry.csv', help='Output CSV')

    args = parser.parse_args()
    main(args)
