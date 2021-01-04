import math
from math import cos, sin
from os import path

import cv2
import numpy as np

# cascades_dir = path.normpath(path.join(cv2.__file__, '..', '..', '..', '..', 'share', 'OpenCV', 'haarcascades'))
# max_size = 720


def rotate(img_in_frame, img, deg):
    rows, cols, _ = img.shape
    hypot = int(math.ceil(math.hypot(rows, cols)))
    M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
    rotated = cv2.warpAffine(img_in_frame, M, (hypot, hypot))
    return rotated


def gray_in_frame(img):
    shape = img.shape
    rows, cols, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hypot = int(math.ceil(math.hypot(rows, cols)))
    frame = np.zeros((hypot, hypot), np.uint8)
    frame[
        math.ceil((hypot - rows) * 0.5) : math.ceil((hypot + rows) * 0.5),
        math.ceil((hypot - cols) * 0.5) : math.ceil((hypot + cols) * 0.5),
    ] = gray
    return frame


def detect(img):
    cascade_f = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    cascade_e = cv2.CascadeClassifier("haarcascade_eye.xml")
    # resize if learch image
    shape = img.shape
    # if max(shape[0], shape[1]) > max_size:
    #     l = max(shape[0], shape[1])
    #     img = cv2.resize(img, (shape[1] * max_size / l, shape[0] * max_size / l))

    rows, cols, _ = img.shape
    hypot = int(math.ceil(math.hypot(rows, cols)))

    frame = gray_in_frame(img)

    def translate(coord, deg):
        x, y = coord
        rad = math.radians(deg)
        return {
            "x": (
                cos(rad) * x
                + sin(rad) * y
                - hypot * 0.5 * cos(rad)
                - hypot * 0.5 * sin(rad)
                + hypot * 0.5
                - (hypot - cols) * 0.5
            )
            / float(cols)
            * 100.0,
            "y": (
                -sin(rad) * x
                + cos(rad) * y
                + hypot * 0.5 * sin(rad)
                - hypot * 0.5 * cos(rad)
                + hypot * 0.5
                - (hypot - rows) * 0.5
            )
            / float(rows)
            * 100.0,
        }

    # rotate and detect faces
    results = []
    for deg in range(-48, 49, 6):
        # M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
        # rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        rotated = rotate(frame, img, deg)
        faces = cascade_f.detectMultiScale(rotated, 1.08, 2)

        # print deg, len(faces)
        for face in faces:
            x, y, w, h = face
            # print(x, y, w, h)
            # eyes in face?
            y_offset = int(h * 0.1)
            roi = rotated[y + y_offset : y + h, x : x + w]
            eyes = cascade_e.detectMultiScale(roi, 1.05)
            eyes = list(
                filter(
                    lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2)
                    and e[1] + e[3] < h / 2,
                    eyes,
                )
            )
            if len(eyes) == 2 and abs(eyes[0][0] - eyes[1][0]) > w / 4:
                score = math.atan2(
                    abs(eyes[1][1] - eyes[0][1]), abs(eyes[1][0] - eyes[0][0])
                )
                if eyes[0][1] == eyes[1][1]:
                    score = 0.0
                results.append(
                    {
                        "center": translate([x + w * 0.5, y + h * 0.5], -deg),
                        "w": float(w) / float(cols) * 100.0,
                        "h": float(h) / float(rows) * 100.0,
                        "r_w": w,
                        "r_h": h,
                        "deg": deg,
                        "y": y,
                        "x": x,
                        "roi_y": (y + y_offset, y + h),
                        "roi_x": x + w,
                        "eyes": [
                            translate(
                                [
                                    x + e[0] + e[2] * 0.5,
                                    y + y_offset + e[1] + e[3] * 0.5,
                                ],
                                -deg,
                            )
                            for e in eyes
                        ],
                        "score": score,
                    }
                )
    # unify duplicate faces
    faces = []
    for result in results:
        x, y = result["center"]["x"], result["center"]["y"]
        exists = False
        for i in range(len(faces)):
            face = faces[i]
            if (
                face["center"]["x"] - face["w"] * 0.5
                < x
                < face["center"]["x"] + face["w"] * 0.5
                and face["center"]["y"] - face["h"] * 0.5
                < y
                < face["center"]["y"] + face["h"] * 0.5
            ):
                exists = True
                if result["score"] < face["score"]:
                    faces[i] = result
                    break
        if not exists:
            faces.append(result)
    for face in faces:
        del face["score"]
    return faces, {"height": shape[0], "width": shape[1]}
