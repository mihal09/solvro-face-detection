import pandas as pd
from sklearn.metrics import mean_squared_error
from basic_model import BasicModel
from pyramid_model import PyramidModel
from tensorflow import keras
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
from pyramid_model import PyramidModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="Model type: basic, pyramid")

parser.add_argument('--path', required=True,
                    help="Model path")

dir_images = "data/img_celeba/"

df = pd.read_csv('data.csv')

df_train = df.iloc[:162771]  # 162771
df_val = df.iloc[162771:182638]  # 19867
df_test = df.iloc[182638:]  # 19961

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory=dir_images,
                                                x_col="filename",
                                                y_col=None,
                                                class_mode=None,
                                                target_size=(224, 224),
                                                shuffle=False,
                                                batch_size=16)


def evaluate(model_class):
    y_true = df_val[['nose_scaled_x', 'nose_scaled_y']]

    model = model_class.get_model()
    custom_generator = model_class.modify_generator(val_generator)

    y_model = model.predict(custom_generator, steps=np.ceil(len(df_val) / 16))
    y_model = model_class.transform_prediction(y_model)[:len(y_true)]

    mse = mean_squared_error(y_true, y_model)
    print("model loss: ", mse)
    return mse


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        if args.model == 'basic':
            model_class = BasicModel()
        elif args.model == 'pyramid':
            model_class = PyramidModel()
        else:
            raise Exception("available model types: basic, pyramid")

        model_class.load_model(args.path)
        evaluate(model_class)
    except Exception as e:
        print(e)


