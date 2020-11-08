import cv2
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import preprocess_input

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help="Path to the model")


def predict_video(model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        height, width, _ = frame.shape

        frame_resized = cv2.resize(frame, (224, 224))
        frame_preprocessed = preprocess_input(frame_resized).reshape(1, 224, 224, 3)

        x_scaled, y_scaled = model.predict(frame_preprocessed)[0]
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
        model = keras.models.load_model(args.model)
        predict_video(model)
    except Exception as e:
        print(e)
