from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model


def build_model(x_shape, y_shape):
    x = Input(shape=x_shape)

    n_filters = 16
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same')(x)
    # conv1 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    # n_filters *= 2
    conv2 = Conv2D(n_filters, 3, activation='relu', padding='same')(pool1)
    # conv2 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=2)(conv2)

    # n_filters *= 2
    conv3 = Conv2D(n_filters, 3, activation='relu', padding='same')(pool2)
    # conv3 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv3)
    # pool3 = MaxPooling2D(pool_size=2)(conv3)

    # n_filters *= 2
    # conv4 = Conv2D(n_filters, 3, activation='relu', padding='same')(pool3)
    # conv4 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv4)
    # pool4 = MaxPooling2D(pool_size=2)(conv4)

    # n_filters *= 2
    # conv5 = Conv2D(n_filters, 3, activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv5)

    # n_filters //= 2
    # up5 = UpSampling2D(size=2)(conv5)
    # up5 = Concatenate()([up5, conv4])
    # conv6 = Conv2D(n_filters, 3, activation='relu', padding='same')(up5)
    # conv6 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv6)

    # n_filters //= 2
    # up6 = UpSampling2D(size=2)(conv6)
    # up6 = Concatenate()([up6, conv3])
    # conv7 = Conv2D(n_filters, 3, activation='relu', padding='same')(up6)
    # conv7 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv7)

    # n_filters //= 2
    up7 = UpSampling2D(size=2)(conv3)  # (conv7)
    up7 = Concatenate()([up7, conv2])
    conv8 = Conv2D(n_filters, 3, activation='relu', padding='same')(up7)
    conv8 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv8)

    # n_filters //= 2
    up8 = UpSampling2D(size=2)(conv8)
    up8 = Concatenate()([up8, conv1])
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same')(up8)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same')(conv9)

    y = Conv2D(y_shape[-1], 3, activation='sigmoid', padding='same')(conv9)

    return Model(x, y)
