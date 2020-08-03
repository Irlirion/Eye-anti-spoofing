# import the necessary packages
import os
from collections import OrderedDict

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
from stabilizer import Stabilizer


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if rects:
        rect = rects[0]

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[36:42]]))
        padding = int(0.2 * w)

        # print(np.array([shape[36:42]]))

        # if prev_shape is None:
        #     prev_shape = [w, h]
        # if abs(prev_shape[0] - w) < 10:
        #     w = prev_shape[0]
        # else:
        #     prev_shape[0] = w
        # if abs(prev_shape[1] - h) < 10:
        #     h = prev_shape[1]
        # else:
        #     prev_shape[1] = h
        right_eye = image[y - padding:y + h + padding, x - padding:x + w + padding]
        # print(right_eye.shape)

        (x2, y2, w2, h2) = cv2.boundingRect(np.array([shape[42:48]]))
        left_eye = image[y2 - padding:y2 + h2 + padding, x2 - padding:x2 + w2 + padding]

        left_eye = cv2.resize(left_eye, (30, 20))
        right_eye = cv2.resize(right_eye, (30, 20))

        return left_eye, right_eye


def capture_video(stream, video_type, M, N):
    # load the input image, resize it, and convert it to grayscale
    cap = cv2.VideoCapture(stream)
    for m in range(M):
        for n in range(N):
            ret, frame = cap.read()
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            if not ret:
                print("Err")
                break

            path = f"./img/{video_type}/{stream[-6:-4]}"
            if not os.path.isdir(path):
                os.mkdir(path)
            eyes = detect(frame)
            if eyes is not None:
                left, right = eyes
                cv2.imshow("left", left)
                cv2.imshow("right", right)
                cv2.imshow("image", cv2.resize(frame, (640, 360)))

                cv2.imwrite(f"{path}/{m}_{n}_left.jpg", left)
                cv2.imwrite(f"{path}/{m}_{n}_right.jpg", right)
            else:
                break
            if cv2.waitKey(10) == 27:
                break

        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                print("Err")
                break
    # break


def capture_cam(id):
    cap = cv2.VideoCapture(id)
    while True:
        ret, frame = cap.read()

        if frame is not None:
            eyes = detect(frame)

            if eyes and eyes[0] is not None:
                yield eyes, frame

        if cv2.waitKey(10) == 27:
            break
    cap.release()


if __name__ == '__main__':
    M = 5
    N = 10

    # for fname in glob.glob("/home/f2a/Видео/eyes/*.mp4"):
    #     print(fname)
    #     # fname = "/home/f2a/Видео/Webcam/phone/001.webm"
    #     capture_video(fname, "real2", M, N)

    # fname = "/home/f2a/Видео/eyes/018.mp4"
    # capture_video(fname, "real")

    images = []

    for eyes, frame in capture_cam(1):
        if eyes:
            left, right = eyes

            if len(images) < 5:
                images.append(left)
            else:
                images = []
            cv2.imshow("left", cv2.resize(left, (150, 100)))
            cv2.imshow("right", cv2.resize(right, (150, 100)))
            cv2.imshow("image", frame)
