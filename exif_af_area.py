import glob
import json
import os
import subprocess
import sys

import cv2

output_directory = "output"
os.makedirs(output_directory, exist_ok=True)
for file in sum([glob.glob(x) for x in sys.argv[1:]], []):
    print(file)
    metadata = json.loads(
        subprocess.run(["exiftool", "-json", file], capture_output=True).stdout.decode(
            "UTF-8"
        ),
    )[0]
    af_image_width = metadata["AFImageWidth"]
    af_image_height = metadata["AFImageHeight"]
    af_area_widths = [int(x) for x in metadata["AFAreaWidths"].split()]
    af_area_heights = [int(x) for x in metadata["AFAreaHeights"].split()]
    af_area_x_positions = [int(x) for x in metadata["AFAreaXPositions"].split()]
    af_area_y_positions = [int(x) for x in metadata["AFAreaYPositions"].split()]
    af_points_in_focus = [int(x) for x in str(metadata["AFPointsInFocus"]).split(",")]
    af_points_selected = [int(x) for x in str(metadata["AFPointsSelected"]).split(",")]
    image = cv2.imread(file)
    color = (0, 0, 0)
    for i in af_points_selected:
        x1 = int(af_area_x_positions[i] - af_area_widths[i] / 2 + af_image_width / 2)
        y1 = af_image_height - int(
            af_area_y_positions[i] + af_area_heights[i] / 2 + af_image_height / 2
        )
        x2 = x1 + af_area_widths[i]
        y2 = y1 + af_area_heights[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
    color = (41, 41, 239)  # Scarlet Red #ef2929
    for i in af_points_in_focus:
        x1 = int(af_area_x_positions[i] - af_area_widths[i] / 2 + af_image_width / 2)
        y1 = af_image_height - int(
            af_area_y_positions[i] + af_area_heights[i] / 2 + af_image_height / 2
        )
        x2 = x1 + af_area_widths[i]
        y2 = y1 + af_area_heights[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
    cv2.imwrite(os.path.join(output_directory, os.path.basename(file)), image)
