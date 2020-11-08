import pandas as pd
from sklearn.metrics import mean_squared_error
from model import get_haar_model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np

dir_images = "../data/CelebA/img_celeba/"

df = pd.read_csv('data.csv')

df_train = df.iloc[:162771]  # 162771
df_val = df.iloc[162771:182638]  # 19867
df_test = df.iloc[182638:]  # 19961


haar = get_haar_model()
y_haar = haar.predict_from_df(df_val, data_dir=dir_images)

#model = keras.models.load_model('resnet_model_big_frozen')
#y_model = model.predict_generator(val_generator, steps=np.ceil(len(df_val) / 16))

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory="../data/CelebA/img_celeba/",
                                                x_col="filename",
                                                y_col=["nose_scaled_x", "nose_scaled_y"],
                                                class_mode="raw",
                                                target_size=(224, 224),
                                                shuffle=False,
                                                batch_size=16)

y_true = df_val[['nose_scaled_x', 'nose_scaled_y']]
y_naive = np.ones(y_true.shape) * 0.5


print("naive loss: ", mean_squared_error(y_true, y_naive))
print("haar loss: ", mean_squared_error(y_true, y_haar))
#print("model loss: ", mean_squared_error(y_true, y_model))
