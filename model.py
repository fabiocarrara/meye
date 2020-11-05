import keras.layers as L
from keras.models import Model


def build_model(x_shape, y_shape, config):
    inp = L.Input(shape=x_shape)
    x = inp
    
    n_stages = config.get('num_stages', 2)
    n_conv = config.get('num_conv', 1)
    n_filters = config.get('num_filters', 16)
    grow_mult = config.get('grow_factor', 1)
    up_activation = config.get('up_act', 'relu')
    
    if up_activation == 'lrelu':
        up_activation = L.LeakyReLU()
    else:
        up_activation = L.Activation(up_activation)
    
    intermediate = []
    
    for _ in range(n_conv):
        x = L.Conv2D(n_filters, 3, activation='relu', padding='same')(x)
    
    # downsample path
    for i in range(n_stages):
        intermediate.append(x)
        x = L.MaxPooling2D(pool_size=2)(x)
        n = n_filters * (grow_mult ** i)
        for _ in range(n_conv):
            x = L.Conv2D(n, 3, activation='relu', padding='same')(x)
    
    middle = L.GlobalAveragePooling2D()(x)
    
    # upsample path
    for i in range(n_stages - 1, -1, -1):
        x = L.UpSampling2D(size=2)(x)
        x = L.Concatenate()([x, intermediate.pop()])
        n = n_filters * (grow_mult ** i)
        for _ in range(n_conv):
            x = L.Conv2D(n, 3, padding='same')(x)
            x = up_activation(x)

    # segmentation mask
    out_mask = L.Conv2D(y_shape[-1], 3, activation='sigmoid', padding='same', name='out_mask')(x)
    # metadata tags
    out_tags = L.Dense(2, activation='sigmoid', name='out_tags')(middle)
    
    return Model(inp, [out_mask, out_tags])
    

def build_model_old(x_shape, y_shape):
    x = L.Input(shape=x_shape)

    n_filters = 16
    conv1 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(x)
    # conv1 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv1)
    pool1 = L.MaxPooling2D(pool_size=2)(conv1)

    # n_filters *= 2
    conv2 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(pool1)
    # conv2 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv2)
    pool2 = L.MaxPooling2D(pool_size=2)(conv2)

    # n_filters *= 2
    conv3 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(pool2)
    # conv3 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv3)
    # pool3 = L.MaxPooling2D(pool_size=2)(conv3)

    # n_filters *= 2
    # conv4 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(pool3)
    # conv4 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv4)
    # pool4 = L.MaxPooling2D(pool_size=2)(conv4)

    # n_filters *= 2
    # conv5 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(pool4)
    # conv5 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv5)

    # n_filters //= 2
    # up5 = L.UpSampling2D(size=2)(conv5)
    # up5 = L.Concatenate()([up5, conv4])
    # conv6 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(up5)
    # conv6 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv6)

    # n_filters //= 2
    # up6 = L.UpSampling2D(size=2)(conv6)
    # up6 = L.Concatenate()([up6, conv3])
    # conv7 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(up6)
    # conv7 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv7)

    # n_filters //= 2
    up7 = L.UpSampling2D(size=2)(conv3)  # (conv7)
    up7 = L.Concatenate()([up7, conv2])
    conv8 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(up7)
    conv8 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv8)

    # n_filters //= 2
    up8 = L.UpSampling2D(size=2)(conv8)
    up8 = L.Concatenate()([up8, conv1])
    conv9 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(up8)
    conv9 = L.Conv2D(n_filters, 3, activation='relu', padding='same')(conv9)

    y = L.Conv2D(y_shape[-1], 3, activation='sigmoid', padding='same')(conv9)

    return Model(x, y)
