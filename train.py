import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from model import get_basic_big_model
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
                                                    batch_size=64)

val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory="../data/CelebA/img_celeba/",
                                                x_col="filename",
                                                y_col=["nose_scaled_x", "nose_scaled_y"],
                                                class_mode="raw",
                                                target_size=(224, 224),
                                                batch_size=32)
freeze = True

model, backbone_model = get_basic_big_model(freeze=freeze)
callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=10, verbose=1),
]
model.fit_generator(train_generator,
                    validation_data=val_generator,
                    validation_steps=50,
                    steps_per_epoch=100,
                    epochs=30,
                    callbacks=callbacks
                    )

if freeze:
    train_generator.reset()
    val_generator.reset()

    for layer in backbone_model.layers[-20:]:
        layer.trainable = True

    model.fit_generator(train_generator,
                        validation_data=val_generator,
                        validation_steps=50,
                        steps_per_epoch=100,
                        epochs=30,
                        callbacks=callbacks
                        )

model.save('resnet_model')
