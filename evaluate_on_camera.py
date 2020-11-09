import cv2
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from basic_model import BasicModel
from pyramid_model import PyramidModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="Model type: basic, pyramid")

parser.add_argument('--path', required=True,
                    help="Model path")


def predict_video(model_class):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        height, width, _ = frame.shape

        if type(model_class) == BasicModel:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_preprocessed = preprocess_input(frame_resized).reshape(1, 224, 224, 3)
        else:
            frame_preprocessed = preprocess_input(frame).reshape(1, height, width, 3)
        x_scaled, y_scaled = model_class.predict(frame_preprocessed)[0]

        x = int(x_scaled * width)
        y = int(y_scaled * height)

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


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
        predict_video(model_class)
    except Exception as e:
        print(e)

