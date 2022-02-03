import os
import math
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from functools import partial

# find pupil center
def _get_pupil_position(pmap, datum, x_shape):
    total_mass = tf.reduce_sum(pmap)
    if total_mass > 0:
        shape = tf.shape(pmap)
        h, w = shape[0], shape[1]
        ii, jj = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
        y = tf.reduce_sum(tf.cast(ii, 'float32') * pmap) / total_mass
        x = tf.reduce_sum(tf.cast(jj, 'float32') * pmap) / total_mass
        return tf.stack((y, x))

    if 'roi_x' in datum and 'roi_y' in datum and 'roi_w' in datum:
        roi_x = tf.cast(datum['roi_x'], 'float32')
        roi_y = tf.cast(datum['roi_y'], 'float32')
        half = tf.cast(datum['roi_w'] / 2, 'float32')
        result = tf.stack((roi_y + half, roi_x + half))
    else:  # fallback to center of the image
        result = tf.cast(tf.stack((x_shape[0] / 2, x_shape[1] / 2)), dtype='float32')

    return result


@tf.function
def load_datum(datum, x_shape=(128, 128, 1), augment=False):

    x = tf.io.read_file(datum['filename'])
    y = tf.io.read_file(datum['target'])

    # HWC [0,1] float32
    x = tf.io.decode_image(x, channels=1, dtype='float32', expand_animations=False)
    y = tf.io.decode_image(y, dtype='float32', expand_animations=False)

    shape = tf.cast(tf.shape(x), 'float32')
    h, w = shape[0], shape[1]
    half_wh = tf.stack((w, h)) / 2

    pupil_map = y[:, :, 0]  # R-channel is the pupil map
    pupil_area = tf.reduce_sum(pupil_map)

    pupil_pos_yx = _get_pupil_position(pupil_map, datum, x_shape)

    if not augment:
        s = tf.minimum(tf.cast(x_shape[0], 'float32'), tf.minimum(h, w))
        pupil_pos_xy = pupil_pos_yx[::-1]
        pupil_new_pos_xy = tf.constant([.5, .5]) * s

        crop_xy = pupil_pos_xy - pupil_new_pos_xy  # crop origin
        # find the feasibility region for the top-left corner of a square crop of size s
        crop_min, crop_max = tf.constant((0., 0.)), tf.stack((w - s, h - s))
        crop_xy = tf.clip_by_value(crop_xy, crop_min, crop_max)

        p = tfa.image.translations_to_projective_transforms(-crop_xy)

    else:  # data augmentation
        # random rotation: pick random angle
        theta = tf.random.uniform([], 0, math.pi / 2)
        cos_t = tf.math.cos(theta)
        sin_t = tf.math.sin(theta)

        # random scale: pick random size of crop around the pupil
        # (constrained by the rotation angle and the original image size)
        min_s = 15
        max_s = tf.math.floor(tf.minimum(w, h) / (sin_t + cos_t))
        s = tf.random.normal([], mean=128, stddev=50)
        s = tf.clip_by_value(s, min_s, max_s)

        # find the feasibility region for the top-left corner of a square crop of size s
        crop_lt = tf.stack((s * sin_t, 0))
        crop_rb = tf.stack((w - s * cos_t, h - s * (sin_t + cos_t)))

        # pick a new random position (in the crop space) in which to place the pupil center
        std = 0.2 if (datum['blink'] == 1) else 0.5  # make sure blinking eyes are shown
        pupil_new_pos_yx = tf.random.normal((2,), mean=0.5, stddev=std) * s

        pupil_pos_y, pupil_pos_x = pupil_pos_yx[0], pupil_pos_yx[1]
        pupil_new_pos_y, pupil_new_pos_x = pupil_new_pos_yx[0], pupil_new_pos_yx[1]

        # crop origin (works.. but xy seem swapped, to double check)
        crop_xy = tf.stack((
            pupil_pos_y + pupil_new_pos_x * sin_t - pupil_new_pos_y * cos_t,
            pupil_pos_x - pupil_new_pos_x * cos_t - pupil_new_pos_y * sin_t
        ))

        # ensure crop is inside image
        crop_xy = tf.clip_by_value(crop_xy, crop_lt, crop_rb)

        # compose transformation
        tr1 = tfa.image.translations_to_projective_transforms(half_wh - crop_xy)
        rot = tfa.image.angles_to_projective_transforms(theta, h, w)
        tr2 = tfa.image.translations_to_projective_transforms(-half_wh)
        p = tfa.image.compose_transforms((tr1, rot, tr2))

    x = tfa.image.transform(x, p, output_shape=(s, s))
    y = tfa.image.transform(y, p, output_shape=(s, s))

    # compute how much pupil is left in the image
    new_pupil_map = y[:, :, 0] 
    new_pupil_area = tf.reduce_sum(new_pupil_map)
    eye = (new_pupil_area / pupil_area) if pupil_area > 0 else 0.

    datum_eye = tf.cast(datum['eye'], 'float32')
    datum_blink = tf.cast(datum['blink'], 'float32')
    if datum_eye == 0:  # set noblink if there is no eye
        datum_blink = 0.

    if (datum_eye == 1) & (datum_blink == 0):  # update eye percentage due to crop (if no blink)
        datum_eye = eye

    if tf.math.reduce_any(tf.shape(x)[:2] != x_shape[:2]):
        x = tf.image.resize(x, x_shape[:2])
        y = tf.image.resize(y, x_shape[:2])

    if augment:
        # random flip
        if tf.random.uniform([]) < 0.5:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)

        if tf.random.uniform([]) < 0.5:
            x = tf.image.flip_up_down(x)
            y = tf.image.flip_up_down(y)

        # random brightness, contrast
        contrast_factor = tf.random.normal([], mean=1.0, stddev=0.4)

        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.adjust_contrast(x, contrast_factor)
        x = tf.clip_by_value(x, 0, 1)

    y = y[:, :, :1]
    y2 = tf.stack((datum_eye, datum_blink))

    return x, y, y2


def get_loader(dataframe, batch_size=8, shuffle=False, **kwargs):
    categories = dataframe.exp.values

    dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(partial(load_datum, **kwargs), num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    dataset = dataset.batch(batch_size)

    # pack targets for keras
    def _pack_targets(*ins):
        inputs = ins[0]
        targets = {'mask': ins[1], 'tags': ins[2]}
        return [inputs, targets]

    dataset = dataset.map(_pack_targets, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
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
    dataset['sub'] = dataset['sub'].astype(str)
    return dataset


if __name__ == '__main__':
    dataset = load_datasets(['NN_human_mouse_eyes'])
    loader, categories = get_loader(dataset, batch_size=1, shiffle=False)

    for x, y in loader:
        print(x, y)
        break
