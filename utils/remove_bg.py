#from rembg import remove
import os
import tqdm
import cv2 as cv
import numpy as np

prefixes = ["ISO7000", "ISO7001", "ISO14617"]
input_dir = "templates"
output_dir = "output_bg_removed_templates"

os.makedirs(output_dir, exist_ok=False)

for filename in tqdm.tqdm(os.listdir(input_dir)):
    for prefix in prefixes:
        if filename.startswith(prefix):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGBA)
            # Set white pixels to transparent
            img[np.all(img == [255, 255, 255, 255], axis=-1)] = [0, 0, 0, 0]
            # Then also set the k-border pixels to transparent to remove bounding boxes arrows
            k = 3
            img[:k, :, 3] = 0
            img[-k:, :, 3] = 0
            img[:, :k, 3] = 0
            img[:, -k:, 3] = 0
            cv.imwrite(output_path, img)
    