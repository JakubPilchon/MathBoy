import cv2 as cv
import os
import torch
#from torchvision.transforms import Grayscale, RandomRotation, Compose, ConvertImageDt
from torchvision.transforms import v2
from dataclasses import dataclass
from typing import List, Tuple, NewType
from model_train.train import CharModel
from PIL import Image, ImageChops
import numpy as np

## custom types:
Bounding_boxes = NewType("Bounding_boxes", List[Tuple[int,int,int,int]])
Pillow_image = NewType("Pillow_image", Image.Image)

@dataclass
class Character:
    x: int
    y: int
    w: int
    h: int
    label: str

    def get_points(self) -> Tuple[int,int,int,int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    
    def __repr__(self) -> str:
        return f'<character: "{self.label}" at ({self.x}, {self.y}, {self.x + self.w}, {self.y + self.h})>'


class Preprocessor:
    
    def __init__(self):
        #Constants:
        self.KERNEL = (4,11)

        self.model = CharModel()
        self.model.load_state_dict(torch.load("model.pt"))

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(),
            v2.Resize((32,32))
            #v2.Lambda(lambda )
        ])

    def get_picture(self):
        try:
            img = cv.imread("img.jpg")
            
        except Exception as e:
            print(e)
        
        return img
    
    def get_bounding_boxes(self, image: Pillow_image) -> Bounding_boxes:
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Convert image to Grayscale
        img_pre = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Thresholding technique on image, makes every character black
        _, img_pre = cv.threshold(img_pre, 240, 255, cv.THRESH_OTSU)
        # Crops image, because loaded up image is 4 pixels higher and wider than it should be
        
        # Reverse colors, now characters are white and background is black
        img_pre = cv.bitwise_not(img_pre)

        # makes characters bigger, better for finding countours
        dilated = cv.dilate(img_pre, cv.getStructuringElement(cv.MORPH_RECT, self.KERNEL), iterations=5)

        #finds contours of characters
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for n in contours:
            # finds rectange encompassing character countour
            x,y,w,h = cv.boundingRect(n)
            if w > 30:
                rectangles.append((x,y,w,h))
        
        return rectangles
    
    def get_character(self, images: Pillow_image, rectangles: Bounding_boxes) -> List[Character]:
        

        for n, (x,y,w,h) in enumerate(rectangles):
            char = preprocessed_img[y:y+h, x:x+w]
            print(x, y)
            try:
                #char = cv.resize(char, (32,32), interpolation=cv.INTER_AREA)
                char = self.transforms(char)
                #cv.imshow(f"character{n}", char)
                char = torch.reshape(char, (1,1,32,32))
                char_class = torch.argmax(self.model.forward(char))
                print("class = ",os.listdir("model_train/dataset")[char_class])
            except Exception as e:
                print("ERROR: ", e)
                print("ERROR CHARACTER SHAPE: ",char.shape)

    def get_characters(self, image: Pillow_image, bounding_boxes:Bounding_boxes) -> List[Character]:
        #image = self.transforms(image)
        character_list = []
        for (x, y, w, h) in bounding_boxes:
            character = image.crop((x, y, x+w, y+h))
            character = self.transforms(character)
            char_class = self.model.forward(character)
            char_class = os.listdir("model_train/dataset")[torch.argmax(char_class)]

            character_list.append(Character(x,y,w,h, char_class))

        return character_list

        
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
    image = Image.open("img.jpg").crop((0,0,1240, 690))
    boxes = pre.get_bounding_boxes(image)
    chars = pre.get_characters(image, boxes)

    for char in chars:
        print(char)
    #print(x,y,w,h)scs
    #image = image.convert("L")
    #image = image.crop((x, y, w+x, y+h))
    #image = image.point(lambda p: 0 if p >= 240  else 255)
    #image = ImageChops.invert(image)
    #image.show("character")
    #pre.open_img()