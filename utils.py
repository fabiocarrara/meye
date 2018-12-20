import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import center_of_mass


def compute_metrics(y, thr=None):
    pupil_map = y[:, :, 0]
    glint_map = y[:, :, 1]

    if thr:
        pupil_map = pupil_map > thr
        glint_map = glint_map > thr

    pc = center_of_mass(pupil_map)  # (y-coord, x-coord)
    gc = center_of_mass(glint_map)
    pa = pupil_map.sum()
    ga = glint_map.sum()

    return pc, gc, pa, ga


def visualizable(x, y, alpha=(.5, .5), thr=.5):
    xx = np.tile(x, (1, 1, 3))
    yy = np.concatenate((y, np.zeros_like(x)), axis=2)  # add a zero blue channel
    mask = yy.max(axis=2) > thr
    mask = mask[:, :, None]
    return np.where(mask, alpha[0] * xx + alpha[1] * yy, xx)


def visualize(x, y):
    n_rows = len(x) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 20 * n_rows // 4))

    for xi, yi, ax in zip(x, y, axes.flatten()):
        i = visualizable(xi, yi)
        ax.imshow(i, cmap=plt.cm.gray)
        ax.grid(False)
