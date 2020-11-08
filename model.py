from focal_loss import BinaryFocalLoss
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Model

import cv2
import numpy as np


def get_basic_big_model(freeze=True):
    shape = (224, 224, 3)
    input_layer = keras.layers.Input(shape=shape)
    backbone_model = keras.applications.ResNet50V2(include_top=False, input_shape=shape)
    if freeze:
        for layer in backbone_model.layers:
            layer.trainable = False
    output = backbone_model(input_layer)
    output = GlobalAveragePooling2D()(output)
    output = Flatten()(output)
    output = BatchNormalization()(output)
    output = Dense(256, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(2, activation='sigmoid')(output)
    model = Model(input_layer, output)

    adam = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=adam,
                  loss='mean_squared_error',
                  metrics=[])

    return model, backbone_model


def get_haar_model():
    class HaarModel:
        def __init__(self):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def predict_from_df(self, df, data_dir):
            output = []
            for index, row in df.iterrows():
                img = cv2.imread(data_dir + row['filename'])
                center_x, center_y = self.predict_img(img)
                output.append([center_x, center_y])
            return np.array(output)

        def predict_img(self, img):
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_height, img_width = frame_gray.shape

            faces = self.face_cascade.detectMultiScale(frame_gray)
            if len(faces) == 0:  # if there are no found faces
                center_x = 0.5
                center_y = 0.5
            else:
                x, y, w, h = faces[0]
                center_x = (x + w / 2.0) / img_width
                center_y = (y + h / 2.0) / img_height
            return center_x, center_y

        def predict(self, x):
            images = np.array(x, dtype='uint8')
            n = images.shape[0]
            output = np.zeros((n, 2), dtype='float64')
            for i in range(n):
                center_x, center_y = self.predict_img(images[i])
                output[i, 0] = center_x
                output[i, 1] = center_y
            return output

    return HaarModel()
