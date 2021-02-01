from tensorflow.keras import layers as L
from tensorflow.keras.models import Model


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
        n = round(n_filters * (grow_mult ** i))
        for _ in range(n_conv):
            x = L.Conv2D(n, 3, activation='relu', padding='same')(x)
    
    middle = L.GlobalAveragePooling2D()(x)
    
    # upsample path
    for i in range(n_stages - 1, -1, -1):
        x = L.UpSampling2D(size=2)(x)
        x = L.Concatenate()([x, intermediate.pop()])
        n = round(n_filters * (grow_mult ** i))
        for _ in range(n_conv):
            x = L.Conv2D(n, 3, padding='same')(x)
            x = up_activation(x)

    # segmentation mask
    out_mask = L.Conv2D(y_shape[-1], 3, activation='sigmoid', padding='same', name='mask')(x)
    # metadata tags
    out_tags = L.Dense(2, activation='sigmoid', name='tags')(middle)
    
    return Model(inp, [out_mask, out_tags])
