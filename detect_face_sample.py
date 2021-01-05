# https://symfoware.blog.fc2.com/blog-entry-2413.html
import cv2
import numpy

prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
confidence_limit = 0.5
net = cv2.dnn.readNetFromCaffe(prototxt, model)
image = cv2.imread("face.jpg")
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
)
net.setInput(blob)
detections = net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < confidence_limit:
        continue
    box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
    start_x, start_y, end_x, end_y = box.astype("int")
    text = "{:.2f}%".format(confidence * 100)
    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
    cv2.putText(
        image, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
    )
cv2.imwrite("result.jpg", image)
