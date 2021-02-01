import os

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import load_img
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


# find pupil center
def _get_pupil_position(pmap, datum, x_shape):
    with np.errstate(invalid='raise'):
        try:
            return center_of_mass(pmap)
        except:
            if 'roi_x' in datum and 'roi_y' in datum and 'roi_w' in datum:
                half = datum['roi_w'] / 2
                return datum['roi_y'] + half, datum['roi_x'] + half

            # fallback to center of the image
            return x_shape[0] / 2, x_shape[1] / 2


def load_datum(datum, x_shape, deterministic=True, glint=False, sample_weights=False):
    channels = 2 if glint else 1
    jpeg_thr = 0.0

    x = load_img(datum['filename'], color_mode='grayscale')
    y = load_img(datum['target'])

    w, h = x.size

    pupil_map = np.array(y)[:,:,0] > jpeg_thr  # R channel = pupil
    pupil_area = pupil_map.sum()

    if deterministic:
        pupil_new_pos = np.array([.5, .5])
        s = 128
        pupil_new_pos_yx = (pupil_new_pos * s).astype(int)
        pupil_pos_yx = _get_pupil_position(pupil_map, datum, x_shape)
        oy, ox = pupil_pos_yx - pupil_new_pos_yx
        l, t, r, b = (ox, oy, ox + s, oy + s)  # Left, Upper; Right, Lower
        # try move the window if a corner is outside the image
        dx = -l if l < 0 else w - r if r > w else 0
        dy = -t if t < 0 else h - b if b > h else 0
        l, t, r, b = (l + dx, t + dy, r + dx, b + dy)
        # the image may still be smaller than the crop area, adjusting ...
        crop = (max(0, l), max(0, t), min(w, r), min(h, b))

    else:  # data augmentation
        # random rotation: pick random angle
        angle = np.random.uniform(0, 90)
        x = x.rotate(angle, expand=True)
        y = y.rotate(angle, expand=True)

        # find pupil in rotated image
        pupil_map = np.array(y)[:, :, 0] > jpeg_thr # R channel = pupil
        pupil_area = pupil_map.sum()
        pupil_pos_yx = _get_pupil_position(pupil_map, datum, x_shape)

        # find image corners in rotated image
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]])  # build rotation for -theta (compensate flipped y-axis)
        centered_corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [-w / 2, h / 2], [w / 2, h / 2]])
        rotated_centered_corners = np.dot(centered_corners, rot.T)
        rotated_corners = rotated_centered_corners - rotated_centered_corners.min(axis=0, keepdims=True)

        # random scale: pick random size of crop around the pupil
        # (this is constrained by the rotation angle and the original image size
        min_s = 15
        max_s = np.floor(min(w, h) / (sin_t + cos_t))
        s = np.random.normal(loc=128, scale=50)
        s = np.clip(s, min_s, max_s)

        A, B, C, D = rotated_corners

        # find the feasibility region for the top-left corner of a square crop of size s
        # the region is a rectangle MNOP
        M = A + ((B - A) / w) * sin_t * s
        N = B + ((A - B) / w) * cos_t * s
        O = -s + D + ((C - D) / w) * sin_t * s
        P = -s + C + ((D - C) / w) * cos_t * s
        MNOP = np.stack((M,N,O,P))

        # pick a new random position (in the crop space) in which to place the pupil center
        std = 0.2 if (datum['blink'] == 1) else 0.5  # make sure blinking eyes are shown
        pupil_new_pos_pct = np.random.normal(loc=0.5, scale=std, size=2)
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
    new_pupil_map = np.array(y)[:, :, 0] > jpeg_thr
    new_pupil_area = new_pupil_map.sum()

    # print(new_pupil_area, pupil_area)
    eye = (new_pupil_area / pupil_area) if pupil_area > 0 else 0

    if datum['eye'] == 0:  # set noblink if there is no eye
        datum['blink'] = 0

    if (datum['eye'] == 1) & (datum['blink'] == 0):  # update eye percentage due to crop (if no blink)
        datum['eye'] = eye

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

        # random brightness, contrast, sharpness
        brightness_factor = np.random.normal(loc=1.0, scale=0.4)
        contrast_factor = np.random.normal(loc=1.0, scale=0.4)
        sharpness_factor = np.random.normal(loc=1.0, scale=0.4)

        x = ImageEnhance.Brightness(x).enhance(brightness_factor)
        x = ImageEnhance.Contrast(x).enhance(contrast_factor)
        x = ImageEnhance.Sharpness(x).enhance(sharpness_factor)

    x = np.expand_dims(np.array(x, dtype=np.float32), -1) / 255.0

    y = np.array(y, dtype=np.float32)[:, :, :channels] / 255.0  # keep only red (and green) channels
    y = np.greater(y, jpeg_thr, out=y)  # remove jpeg artifacts in maps
        
    y2 = np.array([datum['eye'], datum['blink']], dtype=np.float32)

    # 5x weight to blinks
    if sample_weights:
        sample_weight2 = 5. if (datum['blink'] == 1) else 1.
        sample_weight = np.full(x_shape, sample_weight2, dtype=np.float32)
        sample_weight2 = np.array(sample_weight2, dtype=np.float32)
        return x, y, y2, sample_weight, sample_weight2

    return x, y, y2


def get_loader(dataframe, x_shape=(128, 128, 1), batch_size=8, deterministic=False, sample_weights=False, shuffle=False):
    categories = dataframe.exp.values

    dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    data_keys = dataset.element_spec.keys()

    if shuffle:
        dataset = dataset.shuffle(1000)

    def _load_datum(*tensors):
        tensors = map(lambda x: x.decode() if isinstance(x, bytes) else x, tensors)
        datum = dict(zip(data_keys, tensors))
        return load_datum(datum, x_shape=x_shape, deterministic=deterministic, sample_weights=sample_weights)

    out_types = [tf.float32, tf.float32, tf.float32]
    if sample_weights:
        out_types += [tf.float32, tf.float32]

    def _wrapped_load_datum(datum):
        sample_values = [datum[k] for k in data_keys]
        return tf.numpy_function(_load_datum, sample_values, out_types)

    dataset = dataset.map(_wrapped_load_datum, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # pack targets for keras
    def _pack_targets(*ins):
        inputs = ins[0]
        targets = {'mask': ins[1], 'tags': ins[2]}
        if len(ins) > 3:
            sample_weight = {'mask': ins[3], 'tags': ins[4]}
            return [inputs, targets, sample_weight]

        return [inputs, targets]

    dataset = dataset.map(_pack_targets)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, categories


def load_datasets(dataset_dirs):

    def _load_and_prepare_annotations(dataset_dir):
        data = os.path.join(dataset_dir, 'annotation', 'annotations.csv')
        data = pd.read_csv(data)
        data['target'] = dataset_dir + '/annotation/png/' + data.filename.str.replace(r'jpe?g', 'png')
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
