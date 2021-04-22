import tensorflow as tf
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
    conv_type = config.get('conv_type', 'conv')
    use_aspp = config.get('aspp', False)

    if up_activation == 'lrelu':
        up_activation = L.LeakyReLU()
    else:
        up_activation = L.Activation(up_activation)

    use_bn = 'bn-' not in conv_type

    conv = L.SeparableConv2D if 'sep-' in conv_type else L.Conv2D
    conv_common = dict(padding='same', use_bias=not use_bn)

    def conv_block(*args, **kwargs):
        def layer(x):
            if use_bn:
                act = kwargs.pop('activation', None)
                x = conv(*args, **kwargs)(x)
                x = L.BatchNormalization()(x)
                return L.Activation(act)(x) if act else x
            return conv(*args, **kwargs)(x)

        return layer

    intermediate = []

    for _ in range(n_conv):
        x = conv_block(n_filters, 3, activation='relu', **conv_common)(x)

    # downsample path
    for i in range(n_stages):
        intermediate.append(x)
        n = round(n_filters * (grow_mult ** i))
        x = conv_block(n, 3, 2, activation='relu', **conv_common)(x)
        for _ in range(n_conv - 1):
            x = conv_block(n, 3, activation='relu', **conv_common)(x)

    middle = L.GlobalAveragePooling2D()(x)

    if use_aspp:
        n = round(n / 4)
        x1 = conv_block(n, 1, dilation_rate=1, activation='relu', **conv_common)(x)
        x2 = conv_block(n, 3, dilation_rate=2, activation='relu', **conv_common)(x)
        x3 = conv_block(n, 3, dilation_rate=4, activation='relu', **conv_common)(x)
        x4 = conv_block(n, 3, dilation_rate=6, activation='relu', **conv_common)(x)

        # global feature
        xg = L.Reshape((1, 1, -1))(middle)
        xg = conv_block(n, 1, activation='relu', **conv_common)(xg)
        feature_tiling = tf.pad(tf.shape(x)[1:3], tf.constant([[1, 1]]), constant_values=1)
        xg = tf.tile(xg, feature_tiling)

        x = tf.concat([x1, x2, x3, x4, xg], axis=-1)

    # upsample path
    for i in range(n_stages - 1, -1, -1):
        x = L.UpSampling2D(size=2, interpolation='bilinear')(x)
        x = L.Concatenate()([x, intermediate.pop()])
        n = round(n_filters * (grow_mult ** i))
        for _ in range(n_conv):
            x = conv_block(n, 3, **conv_common)(x)
            x = up_activation(x)

    # segmentation mask
    out_mask = conv(y_shape[-1], 3, activation='sigmoid', padding='same', name='mask')(x)
    # metadata tags (is_eye and is_blink)
    out_tags = L.Dense(2, activation='sigmoid', name='tags')(middle)

    return Model(inp, [out_mask, out_tags])
