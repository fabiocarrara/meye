import sys
sys.path += ['models/deeplab']

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model, Sequential

from deeplabv3p.models.deeplabv3p_resnet50 import Deeplabv3pResNet50
from deeplabv3p.models.deeplabv3p_mobilenetv3 import Deeplabv3pMobileNetV3Small, Deeplabv3pLiteMobileNetV3Small, Deeplabv3pMobileNetV3Large, Deeplabv3pLiteMobileNetV3Large
from deeplabv3p.models.deeplabv3p_xception import Deeplabv3pXception
from deeplabv3p.models.deeplabv3p_peleenet import Deeplabv3pPeleeNet, Deeplabv3pLitePeleeNet

AVAILABLE_BACKBONES = {
    'resnet50': Deeplabv3pResNet50,
    'xception': Deeplabv3pXception,
    'mobilenetv3-large': Deeplabv3pMobileNetV3Large,
    'lite-mobilenetv3-large': Deeplabv3pLiteMobileNetV3Large,
    'mobilenetv3-small': Deeplabv3pMobileNetV3Small,
    'lite-mobilenetv3-small': Deeplabv3pLiteMobileNetV3Small,
    'peleenet': Deeplabv3pPeleeNet,
    'lite-peleenet': Deeplabv3pLitePeleeNet,
}

AVAILABLE_PRETRAINED_WEIGHTS = {
    'resnet50': 'imagenet',
    'xception': None,  # 'pascalvoc', # needs fix in upstream
    'mobilenetv3-large': 'imagenet',
    'lite-mobilenetv3-large': 'imagenet',
    'mobilenetv3-small': 'imagenet',
    'lite-mobilenetv3-small': 'imagenet',
    'peleenet': 'imagenet',
    'lite-peleenet': 'imagenet',
}

def build_model(input_shape, output_shape, config):

    assert input_shape[:2] == output_shape[:2], "Only same input-output HW shapes are supported."
    num_classes = output_shape[2]

    # backbone pretends RGB images to use pretrained weights
    needs_rgb_conversion = input_shape[2] != 3
    backbone_input_shape = (input_shape[:2] + (3,)) if needs_rgb_conversion else input_shape
    backbone_name = config.get('backbone', 'resnet50')
    weights = config.get('weights', AVAILABLE_PRETRAINED_WEIGHTS[backbone_name])
    backbone_fn = AVAILABLE_BACKBONES[backbone_name]
    backbone, backbone_len = backbone_fn(input_shape=backbone_input_shape, num_classes=num_classes, weights=weights, OS=8)

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
        rgb_input = L.Lambda(lambda x: K.tile(x, (1, 1, 1, 3)) , name='gray2rgb')(gray_input)  # we assume BHWC
        out_mask, out_tags = model(rgb_input)

        # rename outputs
        out_mask = L.Lambda(lambda x: x, name='mask')(out_mask)
        out_tags = L.Lambda(lambda x: x, name='tags')(out_tags)
        model = Model(inputs=gray_input, outputs=[out_mask, out_tags])

    return model


if __name__ == "__main__":
    shape = (128, 128, 1)
    model = build_model(shape, shape, {'weights': None})#, 'backbone': 'lite-mobilenetv3-small'})
    model.summary()
    import pdb; pdb.set_trace()
