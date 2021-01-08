# https://symfoware.blog.fc2.com/blog-entry-2413.html
import glob
import os
import sys

import cv2
import numpy

import detect_blur_for_face

prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
confidence_limit = 0.3
net = cv2.dnn.readNetFromCaffe(prototxt, model)
output_directory = "output"
os.makedirs(output_directory, exist_ok=True)
for file in sum([glob.glob(x) for x in sys.argv[1:]], []):
    print(file)
    image = detect_blur_for_face.resize_image(cv2.imread(file))
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(gray[y1:y2, x1:x2], (64, 64))
        var = detect_blur_for_face.variance_of_laplacian(face_image).var()
        text = f"{int(var)}"
        y = y1 - 20 if y1 - 20 > 50 else y1 + 50
        # Scarlet Red #ef2929 or Chameleon #8ae234
        color = (41, 41, 239) if var < 100 else (52, 226, 138)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
        cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)
    cv2.imwrite(os.path.join(output_directory, os.path.basename(file)), image)
