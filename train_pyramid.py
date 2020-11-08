import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from PyramidModel import PyramidModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

df = pd.read_csv('data.csv')

df_train = df.iloc[:162771]  # 162771
df_val = df.iloc[162771:182638]  # 19867
df_test = df.iloc[182638:]  # 19961

train_datagen = ImageDataGenerator(brightness_range=[0.2, 1.5], preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory="../data/CelebA/img_celeba/",
                                                    x_col="filename",
                                                    y_col=["nose_scaled_x", "nose_scaled_y"],
                                                    class_mode="raw",
                                                    target_size=(224, 224),
                                                    batch_size=16)

val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory="../data/CelebA/img_celeba/",
                                                x_col="filename",
                                                y_col=["nose_scaled_x", "nose_scaled_y"],
                                                class_mode="raw",
                                                target_size=(224, 224),
                                                batch_size=16)


class WrapGenerator:
    def __init__(self, generator):
        self.generator = generator

    def generator_function(self):
        while True:  # Select files (paths/indices) for the batch
            x, y = self.generator.next()
            n = x.shape[0]
            size_x = x.shape[2] // 4
            size_y = x.shape[1] // 4
            y_new = np.zeros((n, size_y, size_x, 1))
            for i in range(n):
                x_pos = int(y[i, 0] * (size_x - 1))
                y_pos = int(y[i, 1] * (size_y - 1))
                y_new[i, y_pos-5:y_pos+5, x_pos-5:x_pos+5, 0] = 1.0
            yield x, y_new


train_generator_wrap = WrapGenerator(train_generator)
val_generator_wrap = WrapGenerator(val_generator)

callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=10, verbose=1),
]

freeze = True

pyramid_model = PyramidModel()
backbone_model = pyramid_model.backbone.backbone_model
train_model = pyramid_model.get_model()

train_model.fit_generator(train_generator_wrap.generator_function(),
                          validation_data=val_generator_wrap.generator_function(),
                          validation_steps=50,
                          steps_per_epoch=100,
                          epochs=50,
                          callbacks=callbacks
                          )

train_model.save('pyramid_model_frozen')

if freeze:
    train_generator.reset()
    val_generator.reset()

    for layer in backbone_model.layers[-20:]:
        layer.trainable = True

    train_model.fit_generator(train_generator_wrap.generator_function(),
                              validation_data=val_generator_wrap.generator_function(),
                              validation_steps=50,
                              steps_per_epoch=100,
                              epochs=20,
                              callbacks=callbacks
                              )

train_model.save('pyramid_model_unfrozen')
