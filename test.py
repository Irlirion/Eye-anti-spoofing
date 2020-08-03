import numpy as np
from dlib2 import capture_cam
from model import get_model
import torch
import cv2


def show_images(frame, eyes):
    if eyes:
        left, right = eyes
        cv2.imshow("left", cv2.resize(left, (150, 100)))
        cv2.imshow("right", cv2.resize(right, (150, 100)))
    cv2.imshow("image", frame)


def test_cam(path):
    images = []
    model = get_model(path, pretrained=True)
    model.eval()
    for eyes, frame in capture_cam(1):
        show_images(frame, eyes)
        if eyes:
            if len(images) == 10:
                sample = torch.tensor(images).float()
                sample.transpose_(0, 1)
                sample.unsqueeze_(0)
                outputs = model(sample)
                print(outputs.item())
                images = []
            image = np.concatenate(eyes)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[np.newaxis, ...]
            images.append(image)

if __name__ == '__main__':
    test_cam("models/eyes_v10.pt")
