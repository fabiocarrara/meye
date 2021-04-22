import sys
sys.path += ['models/deeplab']

import tensorflow as tf

from tensorflow.keras import layers as L
from tensorflow.keras.models import Model, Sequential

from deeplabv3p.models.deeplabv3p_resnet50 import Deeplabv3pResNet50


def build_model(input_shape, output_shape, config):

    assert input_shape[:2] == output_shape[:2], "Only same input-output HW shapes are supported."
    num_classes = output_shape[2]

    # backbone pretends RGB images to use pretrained weights
    needs_rgb_conversion = input_shape[2] != 3
    backbone_input_shape = (input_shape[:2] + (3,)) if needs_rgb_conversion else input_shape
    backbone, backbone_len = Deeplabv3pResNet50(input_shape=backbone_input_shape, num_classes=num_classes, weights='imagenet', OS=8)

    # segmentation mask
    out_mask = backbone.get_layer('pred_resize').output
    out_mask = L.Activation('sigmoid', name='mask')(out_mask)

    # metadata tags (is_eye and is_blink)
    middle = backbone.get_layer('image_pooling').output
    middle = L.Flatten()(middle)
    out_tags = L.Dense(2, activation='sigmoid', name='tags')(middle)

    model = Model(inputs=backbone.input, outputs=[out_mask, out_tags])

    if needs_rgb_conversion:
        gray_input = L.Input(shape=input_shape)
        rgb_input = L.Lambda(tf.image.grayscale_to_rgb)(gray_input)
        out_mask, out_tags = model(rgb_input)

        # rename outputs
        out_mask = L.Lambda(lambda x: x, name='mask')(out_mask)
        out_tags = L.Lambda(lambda x: x, name='tags')(out_tags)
        model = Model(inputs=gray_input, outputs=[out_mask, out_tags])
    
    return model


if __name__ == "__main__":
    shape = (128, 128, 1)
    model = build_model(shape, shape, {'aspp': True})
    model.summary()
    import pdb; pdb.set_trace()