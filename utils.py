import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import center_of_mass, label, sum as area


def nms_on_area(x, s):  # x is a binary image, s is a structuring element
    labels, num_labels = label(x, structure=s)  # find connected components
    if num_labels > 1:
        indexes = np.arange(1, num_labels + 1)
        areas = area(x, labels, indexes)  # compute area for each connected components
        
        biggest = max(zip(areas, indexes))[1]  # get index of largest component
        x[labels != biggest] = 0  # discard other components

    return x


def compute_metrics(y, thr=None, nms=False):
    pupil_map = y[:, :, 0]
    glint_map = y[:, :, 1]

    if thr:
        pupil_map = pupil_map > thr
        glint_map = glint_map > thr
        
        if nms:  # perform non-maximum suppression: keep only largest area
            s = np.ones((3, 3))  # connectivity structure
            pupil_map = nms_on_area(pupil_map, s)
            glint_map = nms_on_area(glint_map, s)

    pc = center_of_mass(pupil_map)  # (y-coord, x-coord)
    gc = center_of_mass(glint_map)
    pa = pupil_map.sum()
    ga = glint_map.sum()

    return pc, gc, pa, ga


def visualizable(x, y, alpha=(.5, .5), thr=0):
    xx = np.tile(x, (3,))  # Gray -> RGB: repeat channels 3 times
    yy = np.concatenate((y, np.zeros_like(x)), axis=-1)  # add a zero blue channel
    mask = yy.max(axis=-1, keepdims=True) > thr  # blend only where a prediction is present
    # mask = mask[:, :, None]
    return np.where(mask, alpha[0] * xx + alpha[1] * yy, xx)


def visualize(x, y, out=None):
    n_rows = len(x) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 20 * n_rows // 4))
    y_masks, y_tags = y
    
    for xi, yi_mask, yi_tags, ax in zip(x, y_masks, y_tags, axes.flatten()):
        i = visualizable(xi, yi_mask)
        ax.imshow(i, cmap=plt.cm.gray)
        ax.grid(False)
        ax.set_title('E: {:.1%} - B: {:.1%}'.format(*yi_tags))
    
    if out:
        plt.savefig(out)
        plt.close()
