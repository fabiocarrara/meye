import glob
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import center_of_mass


def is_contained(ltrb, corners):
    """
        A------B
        |      |
        C------D
    """
    A, B, C, D = corners

    def is_in(X):
        return (np.dot(C - A, X - A) > 0 and
                np.dot(A - C, X - C) > 0 and
                np.dot(B - A, X - A) > 0 and
                np.dot(A - B, X - B) > 0)

    l, t, r, b = ltrb
    lt = np.array((l, t))
    lb = np.array((l, b))
    rt = np.array((r, t))
    rb = np.array((r, b))

    c1 = is_in(lt)
    c2 = is_in(lb)
    c3 = is_in(rt)
    c4 = is_in(rb)

    return c1 and c2 and c3 and c4

def load_xy(datum, x_shape, deterministic=True, no_pad=False):
    x = load_img(datum.filename, color_mode='grayscale')
    y = load_img(datum.target)

    w, h = x.size

    # find pupil center
    def _get_pupil_position(pmap):
        with np.errstate(invalid='raise'):
            try:
                return center_of_mass(pmap)
            except:
                # print('Center of mass not found, defaulting to image center:', datum.target)
                return (x_shape[0] / 2, x_shape[1] / 2)

    pupil_map = np.array(y)[:,:,0]  # R channel = pupil
    pupil_area = pupil_map.sum()

    if deterministic:
        pupil_new_pos = np.array([.5, .5])
        s = 128
        pupil_new_pos_yx = (pupil_new_pos * s).astype(int)
        pupil_pos_yx = _get_pupil_position(pupil_map)
        oy, ox = pupil_pos_yx - pupil_new_pos_yx
        crop = (ox, oy, ox + s, oy + s)  # Left, Upper; Right, Lower

        if no_pad:
            l, t, r, b = crop
            dx = -l if l < 0 else w - r if r > w else 0
            dy = -t if t < 0 else h - b if b > h else 0
            crop = (l + dx, t + dy, r + dx, b + dy)
            # the image may still be smaller than the crop area, adjusting ...
            l, t, r, b = crop
            crop = (max(0, l), max(0, t), min(w, r), min(h, b))

    else:  # random rotation, random pupil position, random scale

        # pick random angle
        angle = np.random.uniform(0, 90)
        x = x.rotate(angle, expand=True)
        y = y.rotate(angle, expand=True)

        # find pupil in rotated image
        pupil_map = np.array(y)[:, :, 0]  # R channel = pupil
        pupil_area = pupil_map.sum()
        pupil_pos_yx = _get_pupil_position(pupil_map)


        # find image corners in rotated image
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]])  # build rotation for -theta (compensate flipped y-axis)
        centered_corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [-w / 2, h / 2], [w / 2, h / 2]])
        rotated_centered_corners = np.dot(centered_corners, rot.T)
        rotated_corners = rotated_centered_corners - rotated_centered_corners.min(axis=0, keepdims=True)

        # pick size of crop around the pupil for scale augmentation
        # (this is constrained by the rotation angle and the original image size
        min_s = 15
        max_s = np.floor(min(w, h) / (sin_t + cos_t))
        s = np.random.normal(loc=128, scale=50)
        s = np.clip(s, min_s, max_s)

        # print(angle, s)

        A, B, C, D = rotated_corners

        # find the feasibility region for the top-left corner of a square crop of size s
        # the region is a rectangle MNOP
        M = A + ((B - A) / w) * sin_t * s
        N = B + ((A - B) / w) * cos_t * s
        O = -s + D + ((C - D) / w) * sin_t * s
        P = -s + C + ((D - C) / w) * cos_t * s
        MNOP = np.stack((M,N,O,P))

        # pick a new random position (in the crop space) in which to place the pupil center
        pupil_new_pos_pct = np.random.normal(loc=0.5, scale=0.2, size=2)
        pupil_new_pos_yx = (pupil_new_pos_pct * s).astype(int)
        crop_top, crop_left = pupil_pos_yx - pupil_new_pos_yx
        OC = np.array([crop_left, crop_top])

        # ensure the crop origin is in the feasible region (MNOP)
        ## we do this in the feasibility region coordinate system (if xy in [0,1]^2, the crop is good):
        ## we first translate to M as new origin
        MNOP_ = MNOP - M
        OC_ = OC - M
        M_,N_,O_,P_ = MNOP_

        ## the we use MN and MP as new basis
        feasible2img = np.array([N_,P_]).T
        img2feasible = np.linalg.inv(feasible2img)
        OC_ = np.dot(img2feasible, OC_)

        ## we apply constraints in the new space and transform back
        OC_ = np.clip(OC_, 0, 1)
        crop_left, crop_top = np.dot(feasible2img, OC_) + M

        crop = (crop_left, crop_top, crop_left + s, crop_top + s)
        
    x = x.crop(crop)
    y = y.crop(crop)

    # compute how much pupil is left in the image
    new_pupil_map = np.array(y)[:, :, 0]
    new_pupil_area = new_pupil_map.sum()

    # print(new_pupil_area, pupil_area)
    eye = (new_pupil_area / pupil_area) if pupil_area > 0 else 0

    if datum.eye & ~datum.blink:
        datum.eye = eye

    x = x.resize(x_shape[:2])  # TODO: check interpolation type
    y = y.resize(x_shape[:2])

    if not deterministic:
        # random flip
        if np.random.rand() < .5:
            x = x.transpose(Image.FLIP_LEFT_RIGHT)
            y = y.transpose(Image.FLIP_LEFT_RIGHT)

        if np.random.rand() < .5:
            x = x.transpose(Image.FLIP_TOP_BOTTOM)
            y = y.transpose(Image.FLIP_TOP_BOTTOM)

        brightness_factor = np.random.normal(loc=1.0, scale=0.4)
        contrast_factor = np.random.normal(loc=1.0, scale=0.4)
        sharpness_factor = np.random.normal(loc=1.0, scale=0.4)

        x = ImageEnhance.Brightness(x).enhance(brightness_factor)
        x = ImageEnhance.Contrast(x).enhance(contrast_factor)
        x = ImageEnhance.Sharpness(x).enhance(sharpness_factor)

    x = np.expand_dims(np.array(x), -1) / 255.0
    y = np.array(y)[:, :, :2] / 255.0  # keep only red and green channels
    y = y > 0.5
        
    y2 = datum[['eye', 'blink']]

    # print('C = {}, crop = {}, s = {}, pnp = {}: '.format(rotated_corners, crop, s, pupil_new_pos), end='')
    # print('E: {} B: {}'.format(datum.eye, datum.blink))

    return pd.Series({'x': x, 'y': y, 'y2': y2})


class DataGen(keras.utils.Sequence):

    def __init__(self, data, data_dir=None, x_shape=(128, 128, 1), batch_size=8, deterministic=False, no_pad=False):
        self.data = data.copy()
        self.data_dir = data_dir
        
        if data_dir is not None:
            self.data['target'] = data_dir + '/annotation/png/' + self.data.filename.str.replace(r'jpe?g', 'png')
            self.data['filename'] = data_dir + '/fullFrames/' + self.data.filename

        self.x_shape = x_shape
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.no_pad = no_pad

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        self.batch = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_xy = self.batch.apply(lambda row: load_xy(row, self.x_shape, deterministic=self.deterministic, no_pad=self.no_pad), axis=1)
        batch_x = np.stack(batch_xy.x)
        batch_y = np.stack(batch_xy.y)
        batch_y2 = np.stack(batch_xy.y2)
        
        return batch_x, [batch_y, batch_y2]


def load_datasets(dataset_dirs):

    def _load_and_prepare_annotations(dataset_dir):
        data = os.path.join(dataset_dir, 'annotation', 'annotations.csv')
        data = pd.read_csv(data)
        data['target'] = dataset_dir + '/annotation/png/' + data.filename # .str.replace(r'jpe?g', 'png')
        data['filename'] = dataset_dir + '/fullFrames/' + data.filename
        return data

    dataset = pd.concat([_load_and_prepare_annotations(d) for d in dataset_dirs])
    return dataset


if __name__ == '__main__':

    x = plt.imread('data/NN_fullframe/fullFrames/y-00B-1-0.jpeg')
    y = plt.imread('data/NN_fullframe/pngs/y-00B-1-0.png')

    x = np.stack([x, ] * 3, axis=2) / 255.
    # plt.figure(figsize=(15, 15))
    plt.imshow(.5 * x + .5 * y, cmap=plt.cm.gray)
    plt.grid(False)
