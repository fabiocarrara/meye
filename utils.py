import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass, label, sum as area


def nms_on_area(x, s):  # x is a binary image, s is a structuring element
    labels, num_labels = label(x, structure=s)  # find connected components
    if num_labels > 1:
        indexes = np.arange(1, num_labels + 1)
        areas = area(x, labels, indexes)  # compute area for each connected components
        
        biggest = max(zip(areas, indexes))[1]  # get index of largest component
        x[labels != biggest] = 0  # discard other components

    return x


def compute_metrics(p, thr=None, nms=False):
    p = p.squeeze()

    if thr:
        p = p > thr
        if nms:  # perform non-maximum suppression: keep only largest area
            s = np.ones((3, 3))  # connectivity structure
            p = nms_on_area(p, s)

    center = center_of_mass(p)
    area = p.sum()
    return center, area


def visualizable(x, y, alpha=(.5, .5), thr=0):
    xx = np.tile(x, (3,))  # Gray -> RGB: repeat channels 3 times
    yy = (y, ) + (np.zeros_like(x),) * (3 - y.shape[-1])
    yy = np.concatenate(yy, axis=-1)  # add a zero channels to pad to RGB
    mask = yy.max(axis=-1, keepdims=True) > thr  # blend only where a prediction is present
    # mask = mask[:, :, None]
    return np.where(mask, alpha[0] * xx + alpha[1] * yy, xx)


def draw_predictions(image, predictions, thr=0):
    x = image.convert('RGBA')

    maps, tags = predictions
    maps = maps[0] if maps.ndim == 4 else maps
    eye, blink = tags.squeeze()
    alpha = maps.max(axis=-1, keepdims=True) > thr

    n_pad = 3 - maps.shape[-1]
    zero_channels = np.zeros(image.size + (n_pad,))
    y = np.concatenate((maps, zero_channels, alpha), axis=-1)  # add pad and masked alpha channel
    y = (y * 255).astype(np.uint8)
    y = Image.fromarray(y).convert('RGBA')

    preview = Image.alpha_composite(x, y)
    draw = ImageDraw.Draw(preview)
    draw.text((5, 5), 'E: {: >3.1%}  B:{: >3.1%}'.format(eye, blink), fill=(0, 0, 255))
    # draw.text((5, image.height - 5), ''.format(blink), fill=(255, 0, 0))

    return preview


def visualize(x, y, out=None, thr=0, n_cols=4, width=20):
    n_rows = len(x) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, width * n_rows // n_cols))
    y_masks, y_tags = y

    axes = axes.flatten() if isinstance(axes, np.ndarray) else (axes,)
    
    for xi, yi_mask, yi_tags, ax in zip(x, y_masks, y_tags, axes):
        i = visualizable(xi, yi_mask, thr=thr)
        ax.imshow(i, cmap=plt.cm.gray)
        ax.grid(False)
        if len(yi_tags) == 2:
            title = 'E: {:.1%} - B: {:.1%}'
        elif len(yi_tags) == 4:
            title = 'pE: {:.1%} - pB: {:.1%} - gtE: {:.1%} - gtB: {:.1%}'

        ax.set_title(title.format(*yi_tags))
    
    if out:
        plt.savefig(out, bbox_inches='tight')
        plt.close()
