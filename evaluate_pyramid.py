import pandas as pd
from sklearn.metrics import mean_squared_error
from model import get_haar_model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
from PyramidModel import PyramidModel

dir_images = "../data/CelebA/img_celeba/"

df = pd.read_csv('data.csv')

df_train = df.iloc[:162771]  # 162771
df_val = df.iloc[162771:182638]  # 19867
df_test = df.iloc[182638:]  # 19961

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory="../data/CelebA/img_celeba/",
                                                x_col="filename",
                                                y_col=None,
                                                class_mode=None,
                                                target_size=(224, 224),
                                                shuffle=False,
                                                batch_size=16)

pyramid_model = PyramidModel()
pyramid_model.load_model('pyramid_model_frozen')

y_model = pyramid_model.get_model().predict_generator(val_generator, steps=np.ceil(len(df_val) / 16))
y_model = pyramid_model.transform_prediction(y_model)



y_true = df_val[['nose_scaled_x', 'nose_scaled_y']]
y_naive = np.ones(y_true.shape) * 0.5


print("naive loss: ", mean_squared_error(y_true, y_naive))
print("model loss: ", mean_squared_error(y_true, y_model))
