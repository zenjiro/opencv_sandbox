import glob
import os
import sys

import cv2
import exiftool

output_directory = "output"
os.makedirs(output_directory, exist_ok=True)
for file in sum([glob.glob(x) for x in sys.argv[1:]], []):
    print(file)
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(file)
        valid_af_points = metadata["MakerNotes:ValidAFPoints"]
        af_area_widths = [int(x) for x in metadata["MakerNotes:AFAreaWidths"].split()]
        af_area_heights = [int(x) for x in metadata["MakerNotes:AFAreaHeights"].split()]
        af_area_x_positions = [int(x) for x in metadata["MakerNotes:AFAreaXPositions"].split()]
        af_area_y_positions = [int(x) for x in metadata["MakerNotes:AFAreaYPositions"].split()]
        af_points_in_focus = [int(x) for x in metadata["MakerNotes:AFPointsInFocus"].split()]
        af_points_selected = [int(x) for x in metadata["MakerNotes:AFPointsSelected"].split()]
        print(
            metadata["MakerNotes:NumAFPoints"],
            metadata["MakerNotes:ValidAFPoints"],
            metadata["MakerNotes:AFImageWidth"],
            metadata["MakerNotes:AFImageHeight"],
            [int(x) for x in metadata["MakerNotes:AFAreaWidths"].split()],
            [int(x) for x in metadata["MakerNotes:AFAreaHeights"].split()],
            [int(x) for x in metadata["MakerNotes:AFAreaXPositions"].split()],
            [int(x) for x in metadata["MakerNotes:AFAreaYPositions"].split()],
            metadata["MakerNotes:AFPointsInFocus"],
            metadata["MakerNotes:AFPointsSelected"],
        )
    image = cv2.imread(file)
    h, w = image.shape[:2]
    print(h, w)
    x1, y1, x2, y2 = 100, 200, 300, 400
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    color = (52, 226, 138)
    text = "hello こんにちは"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 20)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 20)
    cv2.imwrite(os.path.join(output_directory, os.path.basename(file)), image)
