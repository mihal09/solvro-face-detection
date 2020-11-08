import pandas as pd
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from basic_model import BasicModel
from pyramid_model import PyramidModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="Model type: basic, pyramid")

parser.add_argument('--path', required=True,
                    help="Model path")

df = pd.read_csv('data.csv')

df_train = df.iloc[:162771]  # 162771
df_val = df.iloc[162771:182638]  # 19867
df_test = df.iloc[182638:]  # 19961

train_datagen = ImageDataGenerator(brightness_range=[0.2, 1.5], preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory="data/img_celeba/",
                                                    x_col="filename",
                                                    y_col=["nose_scaled_x", "nose_scaled_y"],
                                                    class_mode="raw",
                                                    target_size=(224, 224),
                                                    batch_size=32)

val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory="data/img_celeba/",
                                                x_col="filename",
                                                y_col=["nose_scaled_x", "nose_scaled_y"],
                                                class_mode="raw",
                                                target_size=(224, 224),
                                                batch_size=32)


def train(model_class, model_path):
    model = model_class.get_model()
    backbone_model = model_class.get_backbone()
    custom_train_generator = model_class.modify_generator(train_generator)
    custom_val_generator = model_class.modify_generator(val_generator)

    callbacks = [
        EarlyStopping(patience=15, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, verbose=1),
    ]

    model.fit_generator(custom_train_generator,
                        validation_data=custom_val_generator,
                        validation_steps=50,
                        steps_per_epoch=100,
                        epochs=30,
                        callbacks=callbacks
                        )

    train_generator.reset()
    val_generator.reset()

    for layer in backbone_model.layers[-20:]:
        layer.trainable = True

    model.fit_generator(custom_train_generator,
                        validation_data=custom_val_generator,
                        validation_steps=50,
                        steps_per_epoch=100,
                        epochs=30,
                        callbacks=callbacks
                        )

    model.save(model_path)


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        if args.model == 'basic':
            model_class = BasicModel()
        elif args.model == 'pyramid':
            model_class = PyramidModel()
        else:
            raise Exception("available model types: basic, pyramid")

        train(model_class, args.path)
    except Exception as e:
        print(e)
