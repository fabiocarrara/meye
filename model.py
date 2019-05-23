from keras.layers import Input, Conv2D, Dense, MaxPooling2D, UpSampling2D, Concatenate, LeakyReLU, Activation, GlobalAveragePooling2D
from keras.models import Model


def build_model(x_shape, y_shape, config):
    inp = Input(shape=x_shape)
    x = inp
    
    n_stages = config.get('num_stages', 2)
    n_conv = config.get('num_conv', 1)
    n_filters = config.get('num_filters', 16)
    grow_mult = config.get('grow_factor', 1)
    up_activation = config.get('up_act', 'relu')
    
    if up_activation == 'lrelu':
        up_activation = LeakyReLU()
    else:
        up_activation = Activation(up_activation)
    
    intermediate = []
    
    for _ in range(n_conv):
        x = Conv2D(n_filters, 3, activation='relu', padding='same')(x)
    
    # downsample path
    for i in range(n_stages):
        intermediate.append(x)
        x = MaxPooling2D(pool_size=2)(x)
        n = n_filters * (grow_mult ** i)
        for _ in range(n_conv):
            x = Conv2D(n, 3, activation='relu', padding='same')(x)
    
    middle = GlobalAveragePooling2D()(x)
    
    # upsample path
    for i in range(n_stages - 1, -1, -1):
        x = UpSampling2D(size=2)(x)
        x = Concatenate()([x, intermediate.pop()])
        n = n_filters * (grow_mult ** i)
        for _ in range(n_conv):
            x = Conv2D(n, 3, padding='same')(x)
            x = up_activation(x)

    # segmentation mask
    out_mask = Conv2D(y_shape[-1], 3, activation='sigmoid', padding='same', name='out_mask')(x)
    # metadata tags
    out_tags = Dense(2, activation='sigmoid', name='out_tags')(middle)
    
    return Model(inp, [out_mask, out_tags])
    

def build_model_old(x_shape, y_shape):
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
