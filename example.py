import cv2
import numpy as np
from keras.models import load_model

from utils import visualizable, compute_metrics


model = load_model('meye-segmentation-2018-12-20.hdf5')

# Region of Interest coordinates
y, x, h, w = (51, 70, 128, 128)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    eye = frame[y:y+h, x:x+w]
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye.astype(np.float32) / 255.0

    eye = eye[:, :, None]  # same as np.expand_dims
    pred = model.predict(eye[None, ...])[0]
    pupil_center, glint_center, pupil_area, glint_area = compute_metrics(pred)
    
    print("pupil-xy=({}, {})\tpupil-area={.2f}\tglint-xy=({},{})\tglint-area={-2f}".format(
        pupil_center[1], pupil_center[0], pupil_area,
        glint_center[1], glint_center[0], glint_area)
    )
    
    # Visualization
    img = visualizable(eye, pred)  # overlays eye image with prediction maps
    img = (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB float -> BGR uint8
    cv2.imshow('preview', img)
    
cap.release()

