import glob
import os
import sys
import json
import subprocess

import cv2
import numpy


def resize_image(image):
    _, width = image.shape[:2]
    while width >= 1500:
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        _, width = image.shape[:2]
    else:
        return image


prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
confidence_limit = 0.3
net = cv2.dnn.readNetFromCaffe(prototxt, model)
output_directory = "output"
os.makedirs(output_directory, exist_ok=True)
for file in sum([glob.glob(x) for x in sys.argv[1:]], []):
    print(file)
    image = resize_image(cv2.imread(file))
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    detected = False
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_limit:
            continue
        detected = True
        box = detections[0, 0, i, 3:7] * numpy.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(gray[y1:y2, x1:x2], (64, 64))
        var = cv2.Laplacian(face_image, cv2.CV_64F).var()
        text = f"{int(var)}"
        # Scarlet Red #ef2929 or Chameleon #8ae234
        color = (41, 41, 239) if var < 100 else (52, 226, 138)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
        cv2.putText(image, text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5)
        cv2.imwrite(os.path.join(output_directory, os.path.basename(file)), image)
    if not detected:
        metadata = json.loads(
            subprocess.run(
                ["exiftool", "-json", file], capture_output=True
            ).stdout.decode("UTF-8"),
        )[0]
        af_image_width = metadata["AFImageWidth"]
        af_image_height = metadata["AFImageHeight"]
        af_area_widths = [int(x) for x in metadata["AFAreaWidths"].split()]
        af_area_heights = [int(x) for x in metadata["AFAreaHeights"].split()]
        af_area_x_positions = [int(x) for x in metadata["AFAreaXPositions"].split()]
        af_area_y_positions = [int(x) for x in metadata["AFAreaYPositions"].split()]
        af_points_in_focus = [
            int(x) for x in str(metadata["AFPointsInFocus"]).split(",")
        ]
        af_points_selected = [
            int(x) for x in str(metadata["AFPointsSelected"]).split(",")
        ]
        scale = width / af_image_width
        color = (41, 41, 239)  # Scarlet Red #ef2929
        min_x1, min_y1, max_x2, max_y2 = af_image_width, af_image_height, 0, 0
        for i in af_points_in_focus:
            x1 = int(
                af_area_x_positions[i] - af_area_widths[i] / 2 + af_image_width / 2
            )
            y1 = af_image_height - int(
                af_area_y_positions[i] + af_area_heights[i] / 2 + af_image_height / 2
            )
            x2 = x1 + af_area_widths[i]
            y2 = y1 + af_area_heights[i]
            min_x1, min_y1, max_x2, max_y2 = (
                min(min_x1, x1),
                min(min_y1, y1),
                max(max_x2, x2),
                max(max_y2, y2),
            )
        cv2.rectangle(
            image,
            (int(min_x1 * scale), int(min_y1 * scale)),
            (int(max_x2 * scale), int(max_y2 * scale)),
            color,
            10,
        )
        cv2.imwrite(os.path.join(output_directory, os.path.basename(file)), image)
