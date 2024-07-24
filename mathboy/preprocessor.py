import cv2 as cv
import os
import torch
from ..model_train import CharModel

class Preprocessor:
    
    def __init__(self):
        # check if file exists 
        assert os.path.isfile('img.jpg')
        # Load up image 
        self.img = cv.imread('img.jpg')
        self.img = self.img[:690, :1250]

        #Constants:
        self.KERNEL = (4,11)

    def get_rectangles(self):
        # Convert image to Grayscale
        img_pre = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # Thresholding technique on image, makes every character black
        _, img_pre = cv.threshold(img_pre, 240, 255, cv.THRESH_OTSU)
        # Crops image, because loaded up image is 4 pixels higher and wider than it should be
        
        # Reverse colors, now characters are white and background is black
        img_pre = cv.bitwise_not(img_pre)

        # makes characters bigger, better for finding countours
        dilated = cv.dilate(img_pre, cv.getStructuringElement(cv.MORPH_RECT, self.KERNEL), iterations=5)

        #finds contours of characters
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for n in contours:
            # finds rectange encompassing character countour
            x,y,w,h = cv.boundingRect(n)
            if w > 30:
                rectangles.append((x,y,w,h))
        
        return rectangles
    
    def get_characters(self, rectangles: list):
        assert isinstance(rectangles, list)

        preprocessed_img =cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        _, preprocessed_img = cv.threshold(preprocessed_img, 240, 255, cv.THRESH_OTSU)
        preprocessed_img = cv.bitwise_not(preprocessed_img)

        for n, (x,y,w,h) in enumerate(rectangles):
            char = preprocessed_img[y:y+h, x:x+w]
            print(char.shape)
            try:
                char = cv.resize(char, (32,32), interpolation=cv.INTER_AREA)
                cv.imshow(f"character{n}", char)
            except Exception as e:
                print("ERROR: ", e)
                print("ERROR CHARACTER SHAPE: ",char.shape)
        k = cv.waitKey(0)
        
    def open_img(self):
        # check if file exists 
        assert os.path.isfile('img.jpg')
        # Load up image 
        img = cv.imread('img.jpg')

        # Convert image to Grayscale
        img_pre = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Thresholding technique on image, makes every character black
        _, img_pre = cv.threshold(img_pre, 250, 255, cv.THRESH_OTSU)
        # Crops image, because loaded up image is 4 pixels higher and wider than it should be
        img_pre = img_pre[:690, :1250]
        # Reverse colors, now characters are white and background is black
        img_pre = cv.bitwise_not(img_pre)

        # makes characters bigger, better for finding countours
        dilated = cv.dilate(img_pre, cv.getStructuringElement(cv.MORPH_RECT,self.KERNEL), iterations=5)

        #finds contours of characters
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        
        for n in contours:
            # finds rectange encompassing character countour
            x,y,w,h = cv.boundingRect(n)
            #if h < 50: h = 50

            if w >= 30:
                # paints green rectangle around character
                cv.rectangle(img, (x,y), (x+w, y+h), color= (0,255,00), thickness= 5)

        cv.imshow("img_pre",dilated)
        cv.imshow("img", img)
        k = cv.waitKey(0)

if __name__ == "__main__":
    pre = Preprocessor()
    pre.open_img()