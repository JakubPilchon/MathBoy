import cv2 as cv
import random as rand
import os
"""
script which adds equal sign class to dataset.
"""

def crop_equal_sign(img: cv.typing.MatLike) -> cv.typing.Rect:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Thresholding technique on image, makes every character black
    _, img = cv.threshold(img, 240, 255, cv.THRESH_OTSU)
        
    # Reverse colors, now characters are white and background is black
    img = cv.bitwise_not(img)

    # makes characters bigger, better for finding countours
    dilated = cv.dilate(img, cv.getStructuringElement(cv.MORPH_RECT, (3,11)), iterations=4)

    #finds contours of characters
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return cv.boundingRect(contours[0]) # returns x,y,w,h



i = 0
for img_path in os.listdir(os.path.join("dataset_before", "eq"))*2:
    img = cv.imread(os.path.join("dataset_before","eq",img_path))

    x,y,w,h = crop_equal_sign(img)
    img = img[y:y+h, x:x+w]
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