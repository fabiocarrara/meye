import glob
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import center_of_mass


def load_xy(datum, x_shape, deterministic=True):
    y = load_img(datum.target)
    if datum[['pupil-y', 'pupil-x']].isnull().values.any():
        pupil_map = img_to_array(y)[:, :, 0]  # R channel = pupil
        with np.errstate(invalid='raise'):
            try:
                pupil_pos_yx = center_of_mass(pupil_map)
            except:
                pupil_pos_yx = (x_shape[0] / 2, x_shape[1] / 2)
    else:
        pupil_pos_yx = datum[['pupil-y', 'pupil-x']].values.astype(int)
    
    # new pupil position: we want it to be in [0.1, 0.9] (in percentage height and width) 95% of times
    pupil_new_pos = np.array([.5, .5]) if deterministic else np.random.normal(loc=0.5, scale=0.2, size=2)

    # size of crop around the pupil (for scale augmentation, needed?)
    s = 128 if deterministic else int(np.random.normal(loc=128, scale=15))

    pupil_new_pos_yx = (pupil_new_pos * s).astype(int)
    oy, ox = pupil_pos_yx - pupil_new_pos_yx

    crop = (ox, oy, ox + s, oy + s)  # Left, Upper; Right, Lower
    x = load_img(datum.filename, color_mode='grayscale').crop(crop).resize(x_shape[:2])  # TODO: check interpolation type
    x = img_to_array(x) / 255.0

    y = y.crop(crop).resize(x_shape[:2])
    y = img_to_array(y)[:, :, :2] / 255.0  # keep only red and green channels
    
    is_pupil_present = y[:, :, 0].sum() > 0
    if datum.eye & ~datum.blink:
        datum.eye = is_pupil_present
        
    y2 = datum[['eye', 'blink']]

    return pd.Series({'x': x, 'y': y, 'y2': y2})


class DataGen(keras.utils.Sequence):

    def __init__(self, data, data_dir=None, x_shape=(128, 128, 1), batch_size=8, deterministic=False):
        self.data = data.copy()
        self.data_dir = data_dir
        
        if data_dir is not None:
            self.data['target'] = data_dir + '/annotation/png/' + self.data.filename.str.replace(r'jpe?g', 'png')
            self.data['filename'] = data_dir + '/fullFrames/' + self.data.filename

        self.x_shape = x_shape
        self.batch_size = batch_size
        self.deterministic = deterministic

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        self.batch = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_xy = self.batch.apply(lambda row: load_xy(row, self.x_shape, deterministic=self.deterministic), axis=1)
        batch_x = np.stack(batch_xy.x)
        batch_y = np.stack(batch_xy.y)
        batch_y2 = np.stack(batch_xy.y2)
        
        return batch_x, [batch_y, batch_y2]


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
