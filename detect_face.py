# https://symfoware.blog.fc2.com/blog-entry-2413.html
import glob
import os
import sys

import cv2
import numpy

prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
confidence_limit = 0.3
confidence_limit2 = 0.5
net = cv2.dnn.readNetFromCaffe(prototxt, model)
output_directory = "output"
os.makedirs(output_directory, exist_ok=True)
for file in sum([glob.glob(x) for x in sys.argv[1:]], []):
    image = cv2.imread(file)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_limit:
            continue
        box = detections[0, 0, i, 3:7] * numpy.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")
        text = "{:.0f}%".format(confidence * 100)
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        # Scarlet Red #ef2929 or Chameleon #8ae234
        color = (41, 41, 239) if confidence < confidence_limit2 else (52, 226, 138)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 20)
        cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 20)
    cv2.imwrite(os.path.join(output_directory, os.path.basename(file)), image)