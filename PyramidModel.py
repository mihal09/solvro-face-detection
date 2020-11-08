import numpy as np
import tensorflow
from tensorflow import keras
from focal_loss import BinaryFocalLoss


class ResNetBackbone:
    def __init__(self):
        if keras.backend.image_data_format() == 'channels_first':
            self.input = keras.layers.Input(shape=(3, None, None))
        else:
            self.input = keras.layers.Input(shape=(None, None, 3))

        self.backbone_model = keras.applications.ResNet50V2(include_top=False, input_tensor=self.input)

        layer_names = {
            'C2': 'conv2_block2_out',
            'C3': 'conv3_block4_out',
            'C4': 'conv4_block6_out',
            'C5': 'conv5_block3_out',
        }

        self.backbone_layers = {shortcut: self.backbone_model.get_layer(layer_name).output
                                for (shortcut, layer_name) in layer_names.items()}


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/image/resize_images .
    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest': tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tensorflow.image.ResizeMethod.BICUBIC,
        'area': tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.compat.v1.image.resize_images(images, size, methods[method], align_corners)


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = tensorflow.transpose(source, (0, 2, 3, 1))
            output = resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = tensorflow.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


def default_regression_model(pyramid_feature_size=256, feature_size=256, output_size=2):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(output_size, name='pyramid_regression', **options)(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs)


def _create_pyramid_features(backbone_layers, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(
        backbone_layers['C5'])
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, backbone_layers['C4']])

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(
        backbone_layers['C4'])
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, backbone_layers['C3']])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(
        backbone_layers['C3'])
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3, backbone_layers['C2']])

    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(
        backbone_layers['C2'])
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    return P2


class PyramidModel:
    def __init__(self, backbone=ResNetBackbone()):
        self.backbone = backbone
        self.input = self.backbone.input

        last_layer = _create_pyramid_features(backbone_layers=self.backbone.backbone_layers, feature_size=256)
        regression_model = default_regression_model(pyramid_feature_size=256, feature_size=256, output_size=1)
        output = regression_model(last_layer)
        self.model = keras.models.Model(inputs=self.input, outputs=output)

        adam = keras.optimizers.Adam(learning_rate=5e-4)
        self.model.compile(optimizer=adam,
                           loss=BinaryFocalLoss(gamma=2.0),
                           metrics=[])

    def get_model(self):
        return self.model

    def predict(self, data):
        y_pred = self.model.predict(data)
        return self.transform_prediction(y_pred)

    def transform_prediction(self, prediction):
        y_pred = prediction
        y_pred = y_pred.reshape(-1, y_pred.shape[1], y_pred.shape[2])
        y_pred_shape = y_pred[0].shape
        outputs = []
        n = y_pred.shape[0]
        for i in range(n):
            y, x = np.unravel_index(np.argmax(y_pred[i], axis=None), y_pred_shape)
            y /= y_pred_shape[0]
            x /= y_pred_shape[1]
            outputs.append([x, y])

        return np.array(outputs)

    def load_model(self, path):
        self.model = keras.models.load_model(path)