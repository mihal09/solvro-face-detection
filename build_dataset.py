import tensorflow as tf
from tensorflow import keras
import pandas as pd
import cv2
import numpy as np
from focal_loss import BinaryFocalLoss

dir_anno = "data/Anno/"
dir_images = "data/img_celeba/"

df = pd.read_csv(dir_anno + "list_landmarks_celeba.txt",
                 header=None,
                 skiprows=2,
                 sep='\s+',
                 names=['filename', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y',
                        'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y'])

df = df[['filename', 'nose_x', 'nose_y']]


def add_width_to_df(row):
    path = dir_images + row['filename']
    width = cv2.imread(path).shape[1]
    return row['nose_x'] / width


def add_height_to_df(row):
    path = dir_images + row['filename']
    height = cv2.imread(path).shape[0]
    return row['nose_y'] / height


# df = df.head(10)
df['nose_scaled_x'] = df.apply(add_width_to_df, axis=1)
df['nose_scaled_y'] = df.apply(add_height_to_df, axis=1)

df.drop(columns=['nose_x', 'nose_y'], inplace=True)
df.to_csv('data.csv')
