import cv2 as cv
import os

"""
dataset preprocess script

dataset source -> https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
unprocessed dataset is contained in model_train/dataset_before directory, while processed dataset is stored in model_train/dataset

PIPELINE:
    1. resize image to 32x32x3 images
    2. reverse colours, so that black is whote and vice versa
    3. save image with standarized name and as a jpg file
"""



for class_name in os.listdir("dataset_before"):
    if class_name.startswith("."): continue
    try:
        os.mkdir(os.path.join("dataset", class_name))
    except FileExistsError:
        print(f"file {class_name} already exists")

    for num, file in enumerate(os.listdir(os.path.join("dataset_before", class_name))):
        if file.startswith("."): continue
        img = cv.imread(os.path.join('dataset_before', class_name, file))
        img = cv.resize(img, (32,32), interpolation=cv.INTER_AREA)
        img = cv.bitwise_not(img)
        cv.imwrite(os.path.join('dataset', class_name, f"{class_name}_{num}.jpg"), img)
