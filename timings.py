import argparse
import time

import numpy as np

from keras.models import load_model
from tqdm import trange


def main(args):

    # x_shape = (args.rh, args.rw, 1)
    # y_shape = (args.rh, args.rw, 2)
    # model = build_model(x_shape, y_shape)
    # model.load_weights('best_weights.hdf5')

    model = load_model(args.model)
    data = np.empty((1, args.rh, args.rw, 1), dtype=np.float32)

    start = time.time()
    for _ in trange(args.n):
        model.predict(data)
    end = time.time()
    elapsed = end - start

    print('Total: {:g} ms ({} ms/img, {} fps)'.format(elapsed, elapsed / args.n, args.n / elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on test video')
    parser.add_argument('model', help='path to model (hdf5)')
    parser.add_argument('-n', type=int, default=100, help='number of predictions')
    parser.add_argument('-rh', type=int, default=128, help='RoI height (-1 for full height)')
    parser.add_argument('-rw', type=int, default=128, help='RoI width (-1 for full width)')

    args = parser.parse_args()
    main(args)
