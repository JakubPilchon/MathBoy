import cv2 as cv
import random as rand
import os
"""
script which adds equal sign class to dataset.
"""

i = 0
for img_path in os.listdir(os.path.join("dataset_before", "eq"))*2:
    img = cv.imread(os.path.join("dataset_before","eq",img_path))
    cv.imshow("eq", img)
    img = cv.resize(img, (28,28), interpolation=cv.INTER_AREA)
    choice = rand.randint(0, 3)

    match choice:
        case 0:
            pass
        case 1:
            img = cv.flip(img, 0)
        case 2:
            img = cv.flip(img, 1)
        case 3:
            img = cv.flip(img, -1)
    img_file = f"eq-{str(i).zfill(4)}.png"

    cv.imwrite(os.path.join("symbols", img_file), img)
    i +=1


print(f"all files saved: {i}")